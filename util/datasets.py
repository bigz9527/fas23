# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance
import random
import numpy as np

import cv2
import scipy.fft as fp
from scipy import signal

from util.DataAugmentation import DataAugmentation



def make_dataset(datalist_path, data_root):
    images = []
    labels = []
    images_path = []

    filelist = None
    with open(datalist_path, 'r') as fp:
        filelist = fp.readlines()
    
    for line in filelist:
        data = line.strip().split(',')
        
        if len(data) > 1: 
            img, label = data[0],data[1]
        else:
            img = data[0]
            label = '0'
        full_path = os.path.join(data_root, img)
        
        images.append(full_path)
        labels.append(int(label))
        images_path.append(img)


    return images, labels, images_path



# from sinet_v2

def pil_loader(path):
    return Image.open(path).convert('RGB')

# several data augumentation strategies
def cv_random_flip(img):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def randomCrop(image):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region)


def randomRotation(image):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
    return image


def colorEnhance(image):
    bright_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

# several data augumentation strategies
def cv_random_resize(img):
    # downsize, then upsize
    resize_flag = random.randint(0, 1)
    if resize_flag == 1 and img.width > 140:
        img = img.resize((int(img.width/2), int(img.height/2)))
    return img


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += np.random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class SUHIFIDataset(data.Dataset):
    def __init__(self, datalist_path, data_root, phase, transform, transform_low_quality, transform_bp, loader=pil_loader):
        images, labels, images_path = make_dataset(datalist_path, data_root)
        self.phase = phase

        self.images = images        
        self.labels = labels
        self.images_path = images_path

        print('all_images cnt: ', len(self.images_path))

        self.transform = transform
        self.transform_low_quality = transform_low_quality
        self.transform_bp = transform_bp

        self.transform_da = DataAugmentation()

        self.loader = loader

    def __getitem__(self, index):
        ret = {}
        
        src_img = self.loader(self.images[index])

        label = self.labels[index]

        image = src_img       

        if self.phase == 'train':
            image = cv_random_resize(image)
            image = randomRotation(image)

        image_cp = image.copy()
        image_bp = cv2.cvtColor(np.asarray(image_cp),cv2.COLOR_RGB2BGR)
        for i in range(image_bp.shape[2]):
            # convert to freq
            freq = fp.fftshift(fp.fft2(image_bp[:, :, i]))
            (w, h) = freq.shape
            cx, cy = int(w / 2), int(h / 2)
            # band-pass
            v1 = signal.gaussian(w, 128)
            v2 = signal.gaussian(h, 128)
            kernel_high = np.outer(v1, v2)
            v1 = signal.gaussian(w, 3)
            v2 = signal.gaussian(h, 3)
            kernel_low = np.outer(v1, v2)
            freq = freq * (kernel_high - kernel_low)
            image_bp[:, :, i] = np.clip(fp.ifft2(fp.ifftshift(freq)).real, 0, 255)

        image_bp = cv2.cvtColor(image_bp, cv2.COLOR_BGR2RGB)

        # transform
        if self.phase == 'train':
            if image.width < 80:
                state = np.random.get_state()
                state_rnd = random.getstate()
                image = self.transform_low_quality(image)
                np.random.set_state(state)
                random.setstate(state_rnd)
                image_bp_t = self.transform_low_quality(Image.fromarray(image_bp))
            else:
                state = np.random.get_state()
                state_rnd = random.getstate()
                # add more data aug 
                image_cv = np.asarray(image)
                image_cv = self.transform_da(image_cv)
                image = self.transform(Image.fromarray(image_cv))
                #image = self.transform(image)
                np.random.set_state(state)
                random.setstate(state_rnd)
                # add more data aug
                image_bp = self.transform_da(image_bp)
                image_bp_t = self.transform(Image.fromarray(image_bp))
        else:
            state = np.random.get_state()
            state_rnd = random.getstate()
            image = self.transform(image)
            np.random.set_state(state)
            random.setstate(state_rnd)
            image_bp_t = self.transform(Image.fromarray(image_bp))

        path = self.images_path[index]
        return image, image_bp_t, label, path

    def __len__(self):
        return len(self.images_path)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    transform_low_quality = build_transform_low_quality(is_train, args)

    transform_bp = build_transform_bp(args)


    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'SUHIFI':
        phase = 'test'
        if is_train:
            phase = 'train'

        dataset = SUHIFIDataset(args.datalist_path, args.data_path, phase=phase, transform=transform, transform_low_quality=transform_low_quality, transform_bp=transform_bp)
        nb_classes = 2

    return dataset, nb_classes

def build_transform_low_quality(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            color_jitter=args.color_jitter,
            auto_augment="original",
            interpolation=args.train_interpolation,
            re_prob=0.0,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    else:
        return None

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=(0.2,1.0),
            ratio=(3./4., 4./3.),
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_transform_bp(args):
    resize_im = args.input_size > 32

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
