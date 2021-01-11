import os
import torch
import shutil
from utils.preproc import Cutout
import torchvision.datasets as dset
import torchvision.transforms as transforms


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def get_imagenet(dataset, data_path, cutout_length, validation):
    dataset = dataset.lower()
    dataset == 'imagenet'
    traindir = data_path + '/train'
    validdir = data_path + '/val'

    CLASSES = 1000
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if cutout_length == 0:
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    elif cutout_length > 0:
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(cutout_length),
            ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    trn_data = train_data
    val_data = valid_data
    input_channels = 3
    input_size = 224
    ret = [input_size, input_channels, CLASSES, trn_data]
    if validation:
        ret.append(val_data)

    return ret
