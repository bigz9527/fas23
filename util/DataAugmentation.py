from skimage.util import random_noise
import random
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np
import math
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch

__alll__ = ["DataAugmentation"]


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

class RandomErasing(ImageOnlyTransform):
    def __init__(
            self,
            probability=0.25, min_area=0.01, max_area=0.05, min_aspect=0.3, max_aspect=None,
            mode='pixel', min_count=1, max_count=3, num_splits=0, device='cpu', always_apply=False):
        super(RandomErasing, self).__init__(always_apply, probability)
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def apply(self, img, **params):
        if len(img.size()) == 3:
            self._erase(img, *img.size(), img.dtype)
        else:
            batch_size, chan, img_h, img_w = img.size()
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(img[i], chan, img_h, img_w, img.dtype)
        return img


class SpeckleNoise(ImageOnlyTransform):
    def __init__(self, mean_min=0.1, mean_max=0.5, clip=True, always_apply=False, p=0.5):
        super(SpeckleNoise, self).__init__(always_apply, p)
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.clip = clip

    def apply(self, img, **params):
        mean = random.random() * (self.mean_max - self.mean_min) + self.mean_min
        img_aug = random_noise(img, mode="speckle", mean=mean, clip=self.clip)
        return (img_aug * 255).astype(np.uint8)


class SaltNoise(ImageOnlyTransform):
    def __init__(self, amount_min=0.0035, amount_max=0.035, clip=True, always_apply=False, p=0.5):
        super(SaltNoise, self).__init__(always_apply, p)
        self.amount_min = amount_min
        self.amount_max = amount_max
        self.clip = clip

    def apply(self, img, **params):
        amount = random.random() * (self.amount_max - self.amount_min) + self.amount_min
        img_aug = random_noise(img, mode="salt", amount=amount, clip=self.clip)
        return (img_aug * 255).astype(np.uint8)


class PepperNoise(ImageOnlyTransform):
    def __init__(self, amount_min=0.02, amount_max=0.2, clip=True, always_apply=False, p=0.5):
        super(PepperNoise, self).__init__(always_apply, p)
        self.amount_min = amount_min
        self.amount_max = amount_max
        self.clip = clip

    def apply(self, img, **params):
        amount = random.random() * (self.amount_max - self.amount_min) + self.amount_min
        img_aug = random_noise(img, mode="pepper", amount=amount, clip=self.clip)
        return (img_aug * 255).astype(np.uint8)


class SPNoise(ImageOnlyTransform):
    def __init__(self, amount_min=0.0035, amount_max=0.035, clip=True, always_apply=False, p=0.5):
        super(SPNoise, self).__init__(always_apply, p)
        self.amount_min = amount_min
        self.amount_max = amount_max
        self.clip = clip

    def apply(self, img, **params):
        amount = random.random() * (self.amount_max - self.amount_min) + self.amount_min
        img_aug = random_noise(img, mode="s&p", amount=amount, clip=self.clip)
        return (img_aug * 255).astype(np.uint8)


class DataAugmentation(object):
    def __init__(self,  noise_p=0.5, blur_p=0.5, compress_p=0.5):
        self.aug = alb.Compose([
            alb.OneOf([
                alb.GaussNoise(var_limit=(10, 60), p=1),
                alb.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 1), p=1),
                #SaltNoise(p=1),
                SpeckleNoise(p=1),
                PepperNoise(p=1),
                SPNoise(p=1),
            ], p=noise_p),
            alb.OneOf([
                alb.Blur(blur_limit=(1, 5), p=1),
                alb.GlassBlur(max_delta=1, p=1),
                alb.MotionBlur(p=1),
                alb.MedianBlur(p=1),
                alb.GaussianBlur(blur_limit=(1, 5), p=1),
                alb.Sharpen(p=1),
                alb.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.12, p=1)
            ], p=blur_p),
            alb.OneOf([
                alb.ImageCompression(quality_lower=40, quality_upper=100, compression_type=0, p=1),
                alb.ImageCompression(quality_lower=40, quality_upper=100, compression_type=1, p=1),
            ], p=compress_p),
        ])

    def __call__(self, img):
        return self.aug(image=img)['image']
