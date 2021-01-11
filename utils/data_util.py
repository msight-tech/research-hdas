# -*- coding: utf-8 -*-
# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
import torchvision.datasets as dset
import utils.preproc as prep


def get_data(dataset, data_path, cutout_length, validation):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = prep.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret
