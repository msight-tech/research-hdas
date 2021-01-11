# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" CNN DAG for architecture search """
from models.get_cell import GetCell
import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast

import genotypes as gt
from models.search_bigDAG import SearchBigDAG, SearchBigDAG_CS


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i: i + len(l)] for i in range(0, len(l_copies), len(l))]
    return l_copies


class SearchStage(nn.Module):
    """
    DAG for search
    Each edge is mixed and continuous relaxed
    """
    def __init__(self, C_in, C, n_classes, n_layers, genotype, n_big_nodes, stem_multiplier):
        """
        C_in: # of input channels
        C: # of starting model channels
        n_classes: # of classes
        n_layers: # of layers
        n_big_nodes: # of intermediate n_cells  # 6
        genotype: the shape of normal cell and reduce cell
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.n_big_nodes = n_big_nodes

        C_cur = stem_multiplier * C  # 4 * 16 = 64
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        self.cells = nn.ModuleList()

        for i in range(n_layers):
            if i in range(n_layers // 3):
                reduction = False
                cell = GetCell(genotype, 4 * C, 4 * C, C, reduction)
                self.cells.append(cell)
            if i in [n_layers // 3]:
                reduction = True
                cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction)
                self.cells.append(cell)
            if i in range(n_layers // 3 + 1, 2 * n_layers // 3):
                reduction = False
                cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction)
                self.cells.append(cell)
            if i in [2 * n_layers // 3]:
                reduction = True
                cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction)
                self.cells.append(cell)
            if i in range(2 * n_layers // 3 + 1, n_layers):
                reduction = False
                cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction)
                self.cells.append(cell)

        self.bigDAG1 = SearchBigDAG(n_big_nodes, self.cells, 0, n_layers // 3, 4 * C)
        self.bigDAG2 = SearchBigDAG(n_big_nodes, self.cells, n_layers // 3 + 1, 2 * n_layers // 3, 8 * C)
        self.bigDAG3 = SearchBigDAG(n_big_nodes, self.cells, 2 * n_layers // 3 + 1, n_layers, 16 * C)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32 * C, n_classes)
    
    def forward(self, x, weights_DAG):
        s0 = s1 = self.stem(x)
        s0 = s1 = self.bigDAG1(s0, s1, weights_DAG[0 * self.n_big_nodes: 1 * self.n_big_nodes])
        s0 = s1 = self.cells[1 * self.n_layers // 3](s0, s1)
        s0 = s1 = self.bigDAG2(s0, s1, weights_DAG[1 * self.n_big_nodes: 2 * self.n_big_nodes])
        s0 = s1 = self.cells[2 * self.n_layers // 3](s0, s1)
        s0 = s1 = self.bigDAG3(s0, s1, weights_DAG[2 * self.n_big_nodes: 3 * self.n_big_nodes])

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class SearchStageController(nn.Module):
    """ SearchDAG controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, genotype, stem_multiplier=4, device_ids=None):
        super().__init__()
        self.n_big_nodes = n_layers // 3
        self.criterion = criterion
        self.genotype = genotype
        self.n_classes = n_classes
        self.n_layers = n_layers
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        n_ops = len(gt.PRIMITIVES2)

        self.alpha_DAG = nn.ParameterList()

        # 3 stages
        for _ in range(3):
            for i in range(self.n_big_nodes):
                # sliding window
                if i < 1:
                    self.alpha_DAG.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
                else:
                    self.alpha_DAG.append(nn.Parameter(1e-3 * torch.randn(3, n_ops)))
        
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        
        self.net = SearchStage(C_in, C, n_classes, n_layers, genotype, self.n_big_nodes, stem_multiplier)
    
    def forward(self, x):
        weights_DAG = [F.softmax(alpha, dim=-1) for alpha in self.alpha_DAG]

        if len(self.device_ids) == 1:
            return self.net(x, weights_DAG)
        
        xs = nn.parallel.scatter(x, self.device_ids)
        wDAG_copies = broadcast_list(weights_DAG, self.device_ids)
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wDAG_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)
    
    def print_alphas(self, logger):
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - DAG")
        for alpha in self.alpha_DAG:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
    
    def DAG(self):
        gene_DAG1 = gt.parse(self.alpha_DAG[0 * self.n_big_nodes: 1 * self.n_big_nodes], k=2)
        gene_DAG2 = gt.parse(self.alpha_DAG[1 * self.n_big_nodes: 2 * self.n_big_nodes], k=2)
        gene_DAG3 = gt.parse(self.alpha_DAG[2 * self.n_big_nodes: 3 * self.n_big_nodes], k=2)

        concat = range(self.n_big_nodes, self.n_big_nodes + 2)

        return gt.Genotype2(DAG1=gene_DAG1, DAG1_concat=concat,
                            DAG2=gene_DAG2, DAG2_concat=concat,
                            DAG3=gene_DAG3, DAG3_concat=concat)
    
    def weights(self):
        return self.net.parameters()
    
    def named_weights(self):
        return self.net.named_parameters()
    
    def alphas(self):
        for n, p in self._alphas:
            yield p
    
    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p


class SearchDistributionDag(SearchStage):
    def __init__(self, C_in, C, n_classes, n_layers, genotype, n_big_nodes, stem_multiplier):
        super().__init__(C_in, C, n_classes, n_layers, genotype, n_big_nodes, stem_multiplier)

        self.bigDAG1 = SearchBigDAG_CS(n_big_nodes, self.cells, 0, n_layers // 3, 4 * C)
        self.bigDAG2 = SearchBigDAG_CS(n_big_nodes, self.cells, n_layers // 3 + 1, 2 * n_layers // 3, 8 * C)
        self.bigDAG3 = SearchBigDAG_CS(n_big_nodes, self.cells, 2 * n_layers // 3 + 1, n_layers, 16 * C)
    
    def forward(self, x, weights_DAG, weights_concat):
        s0 = s1 = self.stem(x)
        s0 = s1 = self.bigDAG1(s0, s1, weights_DAG[0 * self.n_big_nodes: 1 * self.n_big_nodes], weights_concat[0])
        s0 = s1 = self.cells[1 * self.n_layers // 3](s0, s1)
        s0 = s1 = self.bigDAG2(s0, s1, weights_DAG[1 * self.n_big_nodes: 2 * self.n_big_nodes], weights_concat[1])
        s0 = s1 = self.cells[2 * self.n_layers // 3](s0, s1)
        s0 = s1 = self.bigDAG3(s0, s1, weights_DAG[2 * self.n_big_nodes: 3 * self.n_big_nodes], weights_concat[2])

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class SearchDistributionController(SearchStageController):
    def __init__(self, C_in, C, n_classes, n_layers, criterion, genotype, stem_multiplier=4, device_ids=None):
        
        super().__init__(C_in, C, n_classes, n_layers, criterion, genotype, stem_multiplier=stem_multiplier, device_ids=device_ids)
        
        # alpha_concat = beta: for changeable stage length
        self.alpha_concat = nn.ParameterList()
        for _ in range(3):
            self.alpha_concat.append(nn.Parameter(1e-3 * torch.randn(5, 1)))
        
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        
        self.net = SearchDistributionDag(C_in, C, n_classes, n_layers, genotype, self.n_big_nodes, stem_multiplier)
    
    def forward(self, x):
        weights_DAG = [F.softmax(alpha, dim=-1) for alpha in self.alpha_DAG]
        weights_concat = [F.softmax(beta, dim=0) for beta in self.alpha_concat]
        if len(self.device_ids) == 1:
            return self.net(x, weights_DAG, weights_concat)

        xs = nn.parallel.scatter(x, self.device_ids)
        wDAG_copies = broadcast_list(weights_DAG, self.device_ids)
        wConcat_copies = broadcast_list(weights_concat, self.device_ids)
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wDAG_copies, wConcat_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def print_alphas(self, logger):
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - DAG")
        for alpha in self.alpha_DAG:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("# Alpha - Concat")
        for beta in self.alpha_concat:
            logger.info(F.softmax(beta, dim=0))
        logger.info("#####################")

        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
    
    def DAG(self):
        gene_DAG1 = gt.parse(self.alpha_DAG[0 * self.n_big_nodes: 1 * self.n_big_nodes], k=2)
        gene_DAG2 = gt.parse(self.alpha_DAG[1 * self.n_big_nodes: 2 * self.n_big_nodes], k=2)
        gene_DAG3 = gt.parse(self.alpha_DAG[2 * self.n_big_nodes: 3 * self.n_big_nodes], k=2)

        # concat = range(2, 2+self.n_big_nodes)  # concat all intermediate nodes
        # concat = range(2+self.n_big_nodes-2, 2+self.n_big_nodes)
        concat1 = gt.parse_concat(self.alpha_concat[0])
        concat2 = gt.parse_concat(self.alpha_concat[1])
        concat3 = gt.parse_concat(self.alpha_concat[2])

        return gt.Genotype2(DAG1=gene_DAG1, DAG1_concat=concat1,
                            DAG2=gene_DAG2, DAG2_concat=concat2,
                            DAG3=gene_DAG3, DAG3_concat=concat3)
