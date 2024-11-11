import sys
sys.path.append(".")
import os
import six
import lmdb
import pickle
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets as datasets 
from PIL import Image


numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}


def build_transform(is_train, dataset, img_size, to_tensor=True):
    interp_mode = transforms.InterpolationMode.BICUBIC
    if dataset == 'imagenet':
        if is_train:
            transform = [
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                ]
        else:
            transform = []
    else:
        if is_train:
            transform = [
                transforms.Grayscale(1),
                transforms.Resize(img_size, interpolation=interp_mode),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform = [
                transforms.Grayscale(1),
                transforms.Resize(img_size, interpolation=interp_mode),
                transforms.CenterCrop(img_size),
            ]
    if to_tensor: transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    return transform


def build_dataset(is_train, dataset, data_dir, img_size, transform=None):
    if transform is None:
        transform = build_transform(is_train, dataset, img_size)
    split = 'train' if is_train else 'test'

    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(data_dir, train=is_train, transform=transform)
        nb_classes = 10
    if dataset == 'cifar100':
        dataset = datasets.CIFAR100(data_dir, train=is_train, transform=transform)
        nb_classes = 100
    if dataset == 'food101':
        dataset = datasets.Food101(data_dir, split=split, transform=transform)
        nb_classes = 101
    if dataset == 'flowers102':
        dataset = datasets.Flowers102(data_dir, split=split, transform=transform)
        nb_classes = 102
    if dataset == 'pet37':
        split = 'trainval' if is_train else 'test'
        dataset = datasets.OxfordIIITPet(data_dir, split=split, transform=transform)
        nb_classes = 37
    if dataset == 'cars196':
        dataset = datasets.StanfordCars(data_dir, split=split, transform=transform)
        nb_classes = 196
    if dataset == 'imagenet':
        if is_train:
            datadir = os.path.join(data_dir, f'train_gray_{img_size}.lmdb')
        else:
            datadir = os.path.join(data_dir, f'val_gray_{img_size}.lmdb')
        dataset = ImageFolderLMDB(datadir, transform)
        nb_classes = 1000
    
    return dataset, nb_classes


def geometric_transform_th(image: torch.Tensor, mode=None):
    dims = (-2, -1)
    flipud = lambda x: torch.flip(x, dims=[-2])
    
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = torch.rot90(image, dims=dims)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = torch.rot90(image, dims=dims)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = torch.rot90(image, k=2, dims=dims)
    elif mode == 5:
        # rotate 180 degree and flip
        image = torch.rot90(image, k=2, dims=dims)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = torch.rot90(image, k=3, dims=dims)
    elif mode == 7:
        # rotate 270 degree and flip
        image = torch.rot90(image, k=3, dims=dims)
        image = flipud(image)
    
    return image.contiguous()


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, length=None):
        self.db_path = db_path
        with lmdb.open(db_path, subdir=os.path.isdir(db_path), 
                       readonly=True, lock=False, readahead=False, meminit=False) as env:
            with env.begin(write=False) as txn:
                self.length = pickle.loads(txn.get(b'__len__')) if length is None else length
                self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform
        self.target_transform = target_transform

    def open_lmdb(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), 
                             readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
            
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
    
    
class FeatureSet(data.Dataset):
    def __init__(self, data_dict, augment=False, hflip_prob=0, crop_pad=0, device=torch.device('cpu')) -> None:
        super().__init__()
        self.ftrs_p = data_dict['ftrs_p'].to(device)
        self.ftrs_n = data_dict['ftrs_n'].to(device)
        self.labels = data_dict['labels']
        self.hflip_prob = hflip_prob
        self.crop_pad = crop_pad
        self.augment = augment
        
    def __getitem__(self, index):        
        ftr_p = self.ftrs_p[index]
        ftr_n = self.ftrs_n[index]
        label = self.labels[index]
        shape = ftr_p.shape
        
        if self.augment:
            if self.crop_pad > 0:
                ftr_p = F.pad(ftr_p, self.crop_pad)
                ftr_n = F.pad(ftr_n, self.crop_pad)
                params = transforms.RandomCrop.get_params(ftr_p, output_size=(shape[-2], shape[-1]))
                ftr_p = F.crop(ftr_p, *params)
                ftr_n = F.crop(ftr_n, *params)
            
            if torch.rand(1) < self.hflip_prob:
                ftr_p = F.hflip(ftr_p)
                ftr_n = F.hflip(ftr_n)
                # mode = random.randint(0, 7)
                # ftr_p = geometric_transform_th(ftr_p, mode)
                # ftr_n = geometric_transform_th(ftr_n, mode)
                
        return (ftr_p, ftr_n), label
    
    def __len__(self):
        return len(self.labels)
    