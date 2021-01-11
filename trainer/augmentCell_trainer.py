# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils.data_util import get_data
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy

from models.augment_cellcnn import AugmentCellCNN

from utils.data_prefetcher import data_prefetcher
from optimizer.LARSSGD import LARS

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP


class AugmentCellTrainer():
    def __init__(self, config):
        self.config = config

        """ device parameters """
        self.world_size = self.config.world_size
        self.rank = self.config.rank
        self.gpu = self.config.local_rank
        self.distributed = self.config.dist

        """get the train parameters"""
        self.total_epochs = self.config.epochs
        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.global_batch_size = self.world_size * self.train_batch_size

        self.max_lr = self.config.lr * self.world_size

        """construct the whole network"""
        self.resume_path = self.config.resume_path
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.construct_model()

        """save checkpoint path"""
        self.save_epoch = 1
        self.ckpt_path = self.config.path

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = 10
        self.logger = self.config.logger
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.config.path, "tb"))
            self.writer.add_text('config', config.as_markdown(), 0)

    def construct_model(self):
        """get data loader"""
        input_size, input_channels, n_classes, train_data, valid_data = get_data(
            self.config.dataset, self.config.data_path, self.config.cutout_length, validation=True
        )

        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data, num_replicas=self.world_size, rank=self.rank
            )
        else:
            self.train_sampler = None

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=(self.train_sampler is None),
                                                        num_workers=self.config.workers,
                                                        pin_memory=True,
                                                        sampler=self.train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(valid_data,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.config.workers,
                                                        pin_memory=True)
        self.sync_bn = self.config.amp_sync_bn
        self.opt_level = self.config.amp_opt_level
        print(f"sync_bn: {self.sync_bn}")

        """build model"""
        print("init model")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.use_aux = self.config.aux_weight > 0.

        model = AugmentCellCNN(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.use_aux, self.config.genotype)
        
        if self.sync_bn:
            model = apex.parallel.convert_syncbn_model(model)
        self.model = model.to(self.device)

        print("init model end!")

        """ build optimizer """
        print("get optimizer")
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay

        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_epochs)

        """init amp"""
        print("amp init!")
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level=self.opt_level
        )
        if self.distributed:
            self.model = DDP(self.model, delay_allreduce=True)
        print("amp init end!")
    
    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logger.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            self.logger.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")
    
    def save_checkpoint(self, epoch, is_best=False):
        if epoch % self.save_epoch == 0 and self.rank == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'amp': amp.state_dict()
                    }
            # filename = os.path.join(self.ckpt_path, f'{epoch}_ckpt.pth.tar')
            # torch.save(state, filename)
            if is_best:
                best_filename = os.path.join(self.ckpt_path, 'best.pth.tar')
                torch.save(state, best_filename)
    
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_lr = self.optimizer.param_groups[0]['lr']
        
        self.model.train()
        prefetcher = data_prefetcher(self.train_loader)
        X, y = prefetcher.next()
        i = 0
        while X is not None:
            i += 1
            N = X.size(0)
            self.steps += 1

            logits, aux_logits = self.model(X)
            loss = self.criterion(logits, y)

            if self.use_aux:
                loss += self.config.aux_weight * self.criterion(aux_logits, y)
            
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
    
    def val_epoch(self, epoch, printer):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        self.model.eval()

        prefetcher = data_prefetcher(self.valid_loader)
        X, y = prefetcher.next()
        i = 0

        with torch.no_grad():
            while X is not None:
                N = X.size(0)
                i += 1

                logits, _ = self.model(X)

                loss = self.criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
                
                if self.rank == 0 and (i % self.config.print_freq == 0 or i == len(self.valid_loader) - 1):
                    printer(f'Valid: Epoch: [{epoch}][{i}/{len(self.valid_loader)}]\t'
                            f'Step {self.steps}\t'
                            f'Loss {losses.avg:.4f}\t'
                            f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})')
                
                X, y = prefetcher.next()
                
        if self.rank == 0:
            self.writer.add_scalar('val/loss', losses.avg, self.steps)
            self.writer.add_scalar('val/top1', top1.avg, self.steps)
            self.writer.add_scalar('val/top5', top5.avg, self.steps)

            printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg

