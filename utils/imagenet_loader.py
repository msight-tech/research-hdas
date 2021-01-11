# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms


class ImageNetLoader():
    def __init__(self, config):
        self.image_dir = config.data_path
        self.num_replicas = config.world_size
        self.rank = config.rank
        self.distributed = config.dist
        self.resize_size = 224
        self.data_workers = config.workers
        self.CLASSES = 1000
    
    def get_loader(self, stage, batch_size):
        dataset = self.get_dataset(stage)
        if self.distributed and stage in ('train', 'ft'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank
            )
        else:
            self.train_sampler = None
        
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val')),
            num_workers=self.data_workers,
            sampler=self.train_sampler,
            pin_memory=True,
        )
        return data_loader
    
    def get_dataset(self, stage):
        image_dir = self.image_dir + f"{'train' if stage in ('train', 'ft') else 'val'}"
        transform = self.get_transform(stage)
        dataset = datasets.ImageFolder(image_dir, transform=transform)
        return dataset

    def get_transform(self, stage):
        t_list = []
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if stage == 'train':
            t_list = [
                transforms.RandomResizedCrop(self.resize_size),
                transforms.RandomHorizontalFlip(),
                color_jitter,
                transforms.ToTensor(),
                normalize
            ]
        elif stage == 'val':
            t_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]
        else:
            raise KeyError("stage is not in ('train' or 'val')")
        transform = transforms.Compose(t_list)
        return transform
