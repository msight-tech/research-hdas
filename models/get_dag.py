# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from models import ops
import genotypes as gt


class GetStage(nn.Module):
    def __init__(self, DAG, cells, start_p, end_p, C_pp, C_p, C):
        super().__init__()
        self.DAG = DAG
        self.cells = cells
        self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=True)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=True)

        if start_p == 0:
            gene = DAG.DAG1
            self.concat = DAG.DAG1_concat
        elif start_p == len(DAG.DAG1) + 1:
            gene = DAG.DAG2
            self.concat = DAG.DAG2_concat
        elif start_p == len(DAG.DAG1) + len(DAG.DAG2) + 2:
            gene = DAG.DAG3
            self.concat = DAG.DAG3_concat

        self.bigDAG = gt.to_dag(C, gene, False)

        for k in range(start_p, end_p + 1):
            self.bigDAG.append(cells[k])

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for i in range(6):
            edges = self.bigDAG[i]
            cell = self.bigDAG[6 + i]
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(cell(s_cur, s_cur))
        
        # s_out = torch.cat(states[6:], dim=1)
        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        return s_out


class GetStage_img(nn.Module):
    def __init__(self, DAG, cells, start_p, end_p, C_pp, C_p, C):
        super().__init__()
        self.DAG = DAG
        self.cells = cells
        self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=True)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=True)

        if start_p == 0:
            gene = DAG.DAG1
            self.concat = DAG.DAG1_concat
        elif start_p == len(DAG.DAG1) + 1:
            gene = DAG.DAG2
            self.concat = DAG.DAG2_concat
        elif start_p == len(DAG.DAG1) + len(DAG.DAG2) + 2:
            gene = DAG.DAG3
            self.concat = DAG.DAG3_concat

        self.bigDAG = gt.to_dag(C, gene, False)

        for k in range(start_p, end_p + 1): # 4
            self.bigDAG.append(cells[k])
    
    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        states = [s0, s1]
        for i in range(4):
            edges = self.bigDAG[i]
            cell = self.bigDAG[4 + i]
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(cell(s_cur, s_cur))
        
        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        return s_out
