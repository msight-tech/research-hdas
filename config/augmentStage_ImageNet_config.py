# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
from utils.parser import get_parser, parse_gpus, BaseConfig
import genotypes as gt


class AugmentStageImageNetConfig(BaseConfig):
    def build_parser(self):

        parser = get_parser("Augment final model of H^s-DAS config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', type=str, default='imagenet', help='Imagenet')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--lr', type=float, default=0.1, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=250, help='# of training epochs') # 600
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=14, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path prob')
        parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
        parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')

        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--DAG', required=True, help='DAG genotype')

        parser.add_argument('--dist', action='store_true', help='use multiprocess_distributed training')
        parser.add_argument('--local_rank', default=0)
        parser.add_argument('--resume_path', type=str, default=None)
        parser.add_argument('--exclude_bias_and_bn', type=bool, default=True)
        parser.add_argument('--warmup_epochs', type=int, default=10)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # self.data_path = './data/'
        self.data_path = './data/imagenet/'
        self.path = os.path.join('results/augment_Stage/imagenet/', self.name)
        # self.path = os.path.join('augments/imagenet/', self.name)
        self.genotype = gt.from_str(self.genotype)
        self.DAG = gt.from_str(self.DAG)
        self.gpus = parse_gpus(self.gpus)
        self.amp_sync_bn = True
        self.amp_opt_level = "O0"
