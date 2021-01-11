# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from logging import logProcesses
import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy

from models.augment_stage_imagenet import AugmentStageImageNet

from utils.data_prefetcher import data_prefetcher
from optimizer.LARSSGD import LARS
from trainer.augmentStage_trainer import AugmentStageTrainer
from utils.imagenet_loader import ImageNetLoader

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AugmentStageImageNetTrainer(AugmentStageTrainer):
    def __init__(self, config):
        super().__init__(config=config)
        self.warmup_epochs = self.config.warmup_epochs
        self.num_examples = 1281167
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size

    def construct_model(self):
        """get data loader"""
        self.data_ins = ImageNetLoader(self.config)
        self.train_loader = self.data_ins.get_loader('train', self.train_batch_size)
        self.valid_loader = self.data_ins.get_loader('val', self.val_batch_size)

        self.sync_bn = self.config.amp_sync_bn
        self.opt_level = self.config.amp_opt_level
        print(f"sync_bn: {self.sync_bn}")

        """build model"""
        print("init model")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1).to(self.device)
        self.use_aux = self.config.aux_weight > 0.
        model = AugmentStageImageNet(224, 3, self.config.init_channels, 1000, self.config.layers, self.use_aux, self.config.genotype, self.config.DAG)
        if self.sync_bn:
            model = apex.parallel.convert_syncbn_model(model)
        self.model = model.to(self.device)
        print("init model end!")

        """build optimizier"""
        print("get optimizer")
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay
        # LARSSGD
        # exclude_bias_and_bn = self.config.exclude_bias_and_bn
        # params = collect_params([self.model], exclude_bias_and_bn=exclude_bias_and_bn)
        # self.optimizer = LARS(params, lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)
        # SGD
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_epochs, eta_min=0)

        """init amp"""
        print("amp init!")
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level=self.opt_level
        )
        if self.distributed:
            self.model = DDP(self.model, delay_allreduce=True)
        print("amp init end!")
    
    def adjust_learning_rate(self, step):
        """learning rate warm up and decay"""
        max_lr = self.max_lr
        min_lr = 1e-3 * self.max_lr
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        # cur_step = epoch * len(self.train_loader)
        cur_lr = self.optimizer.param_groups[0]['lr']
        
        self.model.train()

        # for step, (X, y) in enumerate(self.train_loader):
        prefetcher = data_prefetcher(self.train_loader)
        X, y = prefetcher.next()
        i = 0
        while X is not None:
            i += 1
            self.adjust_learning_rate(self.steps)
            N = X.size(0)
            self.steps += 1

            logits, aux_logits = self.model(X)

            loss = self.criterion_smooth(logits, y)
            if self.use_aux:
                loss += self.config.aux_weight * self.criterion_smooth(aux_logits, y)
            
            self.optimizer.zero_grad()
            if self.opt_level == 'O0':
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if self.steps % self.log_step == 0 and self.rank == 0:
                self.writer.add_scalar('train/lr', round(cur_lr, 5), self.steps)
                self.writer.add_scalar('train/loss', loss.item(), self.steps)
                self.writer.add_scalar('train/top1', prec1.item(), self.steps)
                self.writer.add_scalar('train/top5', prec5.item(), self.steps)

            if self.gpu == 0 and (i % self.config.print_freq == 0 or i == len(self.train_loader) - 1):
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                    )

            X, y = prefetcher.next()

        if self.gpu == 0:
            printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
