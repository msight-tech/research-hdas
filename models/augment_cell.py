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


class AugmentCell(nn.Module):
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction, layer_id, n_layers):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal1)
        self.layer_id = layer_id

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0)

        if self.layer_id < n_layers // 3:
            gene = genotype.normal1
            self.concat = genotype.normal1_concat
        elif self.layer_id == n_layers // 3:
            gene = genotype.reduce1
            self.concat = genotype.reduce1_concat
        elif self.layer_id < 2 * n_layers // 3:
            gene = genotype.normal2
            self.concat = genotype.normal2_concat
        elif self.layer_id == 2 * n_layers // 3:
            gene = genotype.reduce2
            self.concat = genotype.reduce2_concat
        elif self.layer_id > 2 * n_layers // 3:
            gene = genotype.normal3
            self.concat = genotype.normal3_concat
        
        self.dag = gt.to_dag(C, gene, reduction)
        # print(self.dag)
    
    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        return s_out
