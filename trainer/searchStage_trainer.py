# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
import time
import torch
from torch.cuda import current_blas_handle, device
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from utils.data_util import get_data
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy

from utils.data_prefetcher import data_prefetcher

from models.search_stage import SearchStageController
from models.architect import Architect


class SearchStageTrainer():
    def __init__(self, config) -> None:
        self.config = config

        self.world_size = 1
        self.gpu = self.config.local_rank
        self.save_epoch = 1
        self.ckpt_path = self.config.path

        """get the train parameters"""
        self.total_epochs = self.config.epochs
        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.global_batch_size = self.world_size * self.train_batch_size
        self.max_lr = self.config.w_lr * self.world_size

        """construct the whole network"""
        self.resume_path = self.config.resume_path
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.gpus[0])
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.construct_model()

        self.steps = 0
        self.log_step = 10
        self.logger = self.config.logger
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.path, "tb"))
        self.writer.add_text('config', config.as_markdown(), 0)
    
    def construct_model(self):
        """get data loader"""
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=0, validation=False
        )

        n_train = len(train_data)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.config.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.config.workers,
                                                        pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.config.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.config.workers,
                                                        pin_memory=True)
        
        """build model"""
        print("init model")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        model = SearchStageController(input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, self.config.genotype, device_ids=self.config.gpus)
        self.model = model.to(self.device)
        print("init model end!")

        """build optimizer"""
        print("get optimizer")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.architect = Architect(self.model, self.config.w_momentum, self.config.w_weight_decay)

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
            self.w_optim.load_state_dict(checkpoint['w_optim'])
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
            self.logger.info(f"--> Loaded checkpoint '{model_path}'(epoch {self.start_epoch})")
        
    def save_checkpoint(self, epoch, is_best=False):
        if epoch % self.save_epoch == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'w_optim': self.w_optim.state_dict(),
                     'alpha_optim': self.alpha_optim.state_dict()
                    }
            if is_best:
                best_filename = os.path.join(self.ckpt_path, 'best.pth.tar')
                torch.save(state, best_filename)
    
    def cal_depth(self, alpha, n_nodes, SW):
        """
        SW: sliding window
        """
        assert len(alpha) == n_nodes, "the length of alpha must be the same as n_nodes"
        d = [0, 0]
        for i in range(n_nodes):
            if i < (SW - 2):
                dd = 0
                for j in range(i + 2):
                    dd += alpha[i][j] * (d[j] + 1)
                dd /= (i + 2)
                d.append(dd)
            else:
                dd = 0
                for s, j in enumerate(range(i - 1, i + 2)):
                    dd += alpha[i][s] * (d[j] + 1)
                dd /= SW
                d.append(dd)
        return sum(sum(d) / n_nodes)
    
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

        self.model.print_alphas(self.logger)
        self.model.train()

        prefetcher_trn = data_prefetcher(self.train_loader)
        prefetcher_val = data_prefetcher(self.valid_loader)
        trn_X, trn_y = prefetcher_trn.next()
        val_X, val_y = prefetcher_val.next()
        i = 0
        while trn_X is not None:
            i += 1
            N = trn_X.size(0)
            self.steps += 1

            # architect step (alpha)
            self.alpha_optim.zero_grad()
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim)
            self.alpha_optim.step()

            self.alpha_optim.zero_grad()
            alpha = self.architect.net.alpha_DAG
            self.n_nodes = self.config.layers // 3
            d_depth1 = self.cal_depth(alpha[0 * self.n_nodes: 1 * self.n_nodes], self.n_nodes, 3)
            d_depth2 = self.cal_depth(alpha[1 * self.n_nodes: 2 * self.n_nodes], self.n_nodes, 3)
            d_depth3 = self.cal_depth(alpha[2 * self.n_nodes: 3 * self.n_nodes], self.n_nodes, 3)
            depth_loss = -1.0 * (d_depth1 + d_depth2 + d_depth3)
            depth_loss.backward()
            self.alpha_optim.step()
            
            # child network step (w)
            self.w_optim.zero_grad()
            logits = self.model(trn_X)
            loss = self.model.criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if self.steps % self.log_step == 0:
                self.writer.add_scalar('train/lr', round(cur_lr, 5), self.steps)
                self.writer.add_scalar('train/loss', loss.item(), self.steps)
                self.writer.add_scalar('train/top1', prec1.item(), self.steps)
                self.writer.add_scalar('train/top5', prec5.item(), self.steps)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                        )
            
            trn_X, trn_y = prefetcher_trn.next()
            val_X, val_y = prefetcher_val.next()
        
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

                logits = self.model(X)
                loss = self.criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
                
                if i % self.config.print_freq == 0 or i == len(self.valid_loader) - 1:
                    printer(f'Valid: Epoch: [{epoch}][{i}/{len(self.valid_loader)}]\t'
                            f'Step {self.steps}\t'
                            f'Loss {losses.avg:.4f}\t'
                            f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})')
                
                X, y = prefetcher.next()

        self.writer.add_scalar('val/loss', losses.avg, self.steps)
        self.writer.add_scalar('val/top1', top1.avg, self.steps)
        self.writer.add_scalar('val/top5', top5.avg, self.steps)

        printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg
