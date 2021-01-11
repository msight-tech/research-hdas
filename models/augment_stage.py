# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch.nn as nn
from models.get_cell import GetCell
from models.get_dag import GetStage
from models import ops


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)
    
    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class AugmentStage(nn.Module):
    """" Augmented DAG-CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 DAG, stem_multiplier=4):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
            genotype: the struct of normal cell & reduce cell
            DAG: the struct of big-DAG
        """
        super().__init__()
        self.C_in = C_in
        self.C = C                   # 36
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.DAG = DAG
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1 

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        # C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()

        lenDAG1, lenDAG2, lenDAG3 = len(self.DAG.DAG1), len(self.DAG.DAG2), len(self.DAG.DAG3)

        for i in range(n_layers):
            # if i in [0,1,2,3,4,5]:
            if i in range(lenDAG1):
                reduction = False
                cell = GetCell(genotype, 4 * C, 4 * C, C, reduction) # out 144=4*C
                # 144 144 36  out=144  DAG_out=144*2=288
                self.cells.append(cell)
            # if i in [6]:
            if i in [lenDAG1]:
                reduction = True
                cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction) # out 72*4=288
                # 288, 288, 72 out=288=4*72
                self.cells.append(cell)
            # if i in [7,8,9,10,11,12]:
            if i in range(lenDAG1 + 1, lenDAG1 + 1 + lenDAG2):
                reduction = False
                cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction) # out 288
                # 288, 288, 72, out=72*4=288  DAG_out=288*2=576
                self.cells.append(cell)
            # if i in [13]:
            if i in [lenDAG1 + 1 + lenDAG2]:
                reduction = True
                cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction) # out 144*4=576
                self.cells.append(cell)
            # if i in [14,15,16,17,18,19]:
            if i in range(lenDAG1 + 2 + lenDAG2, lenDAG1 + 2 + lenDAG2 + lenDAG3):
                reduction = False
                cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction) # out 144*4=576
                self.cells.append(cell)  # DAG_out=576*2=1152
        
        self.bigDAG1 = GetStage(DAG, self.cells, 0, lenDAG1 - 1, 4 * C, 4 * C, 4 * C)
        self.bigDAG2 = GetStage(DAG, self.cells, lenDAG1 + 1, lenDAG1 + lenDAG2, 8 * C, 8 * C, 8 * C)
        self.bigDAG3 = GetStage(DAG, self.cells, lenDAG1 + 2 + lenDAG2, lenDAG1 + 1 + lenDAG2 + lenDAG3, 16 * C, 16 * C, 16 * C)

        self.aux_head = AuxiliaryHead(input_size // 4, 16 * C, n_classes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32 * C, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None
        s0 = s1 = self.bigDAG1(s0, s1)
        s0 = s1 = self.cells[6](s0, s1)
        s0 = s1 = self.bigDAG2(s0, s1) 
        s0 = s1 = self.cells[13](s0, s1)
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
