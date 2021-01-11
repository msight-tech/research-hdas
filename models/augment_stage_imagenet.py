# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from os import terminal_size
import torch.nn as nn
from models import ops
from models.get_cell import GetCell
from models.get_dag import GetStage_img


class AuxiliaryHeadImagenet(nn.Module):
    def __init__(self, input_size, C, n_classes) -> None:
        assert input_size == 14
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class AugmentStageImageNet(nn.Module):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 DAG, stem_multiplier=4):
        
        super().__init__()
        self.C_in = C_in
        self.C = C                   # 36
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.DAG = DAG
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        C_cur = stem_multiplier * C
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C_cur, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_cur),
        )
        C_pp, C_p, C_cur = C_cur, C_cur, C
        self.cells = nn.ModuleList()
        lenDAG1, lenDAG2, lenDAG3 = len(self.DAG.DAG1), len(self.DAG.DAG2), len(self.DAG.DAG3)
        
        for i in range(n_layers):
            if i in range(lenDAG1):
                reduction = False
                cell = GetCell(genotype, 4 * C, 4 * C, C, reduction)
                self.cells.append(cell)
            if i in [lenDAG1]:
                reduction = True
                cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction)
                self.cells.append(cell)
            if i in range(lenDAG1 + 1, lenDAG1 + 1 + lenDAG2):
                reduction = False
                cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction)
                self.cells.append(cell)
            if i in [lenDAG1 + 1 + lenDAG2]:
                reduction = True
                cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction)
                self.cells.append(cell)
            if i in range(lenDAG1 + 2 + lenDAG2, lenDAG1 + 2 + lenDAG2 + lenDAG3):
                reduction = False
                cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction)
                self.cells.append(cell)

        self.bigDAG1 = GetStage_img(DAG, self.cells, 0, lenDAG1 - 1, 4 * C, 4 * C, 4 * C)
        self.bigDAG2 = GetStage_img(DAG, self.cells, lenDAG1 + 1, lenDAG1 + lenDAG2, 8 * C, 8 * C, 8 * C)
        self.bigDAG3 = GetStage_img(DAG, self.cells, lenDAG1 + 2 + lenDAG2, lenDAG1 + 1 + lenDAG2 + lenDAG3, 16 * C, 16 * C, 16 * C)

        self.aux_head = AuxiliaryHeadImagenet(14, 16 * C, n_classes)
        self.gap = nn.AvgPool2d(7)
        self.linear = nn.Linear(32 * C, n_classes)
    
    def forward(self, x):
        aux_logits = None
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        s0 = s1 = self.bigDAG1(s1, s1)
        s0 = s1 = self.cells[4](s0, s1)
        s0 = s1 = self.bigDAG2(s0, s1)
        s0 = s1 = self.cells[9](s0, s1)

        aux_logits = self.aux_head(s1)

        s0 = s1 = self.bigDAG3(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits, aux_logits
    
    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
