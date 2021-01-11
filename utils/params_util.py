# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

def collect_params(model_list, exclude_bias_and_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    param_list = []
    for model in model_list:
        for name, param in model.named_parameters():
            if exclude_bias_and_bn and ('bn' in name or 'bias' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
            else:
                param_dict = {'params': param}
            param_list.append(param_dict)
    return param_list