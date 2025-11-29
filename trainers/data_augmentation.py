import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image, ImageFilter

class ColorPreservingAugmentation:
    """保持颜色特征的数据增强类，仅保留旋转、裁剪、噪声和模糊"""
    def __init__(self,
                 rotation_range=(-15, 15),
                 crop_scale=(0.8, 1.0),
                 noise_std=0.01,
                 blur_prob=0.3,
                 blur_radius=(0.1, 2.0)):
        self.rotation_range = rotation_range
        self.crop_scale = crop_scale
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius

    def random_rotation(self, img):
        """随机旋转"""
        angle = random.uniform(*self.rotation_range)
        return img.rotate(angle, resample=Image.BICUBIC, expand=False)

    def random_crop(self, img):
        """随机裁剪"""
        width, height = img.size
        scale = random.uniform(*self.crop_scale)
        new_width = int(width * scale)
        new_height = int(height * scale)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        return img.crop((left, top, left + new_width, top + new_height))

    def add_gaussian_noise(self, img):
        """添加高斯噪声"""
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, self.noise_std * 255, img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def random_blur(self, img):
        """随机模糊"""
        if random.random() < self.blur_prob:
            radius = random.uniform(*self.blur_radius)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = self.random_rotation(img)
        img = self.random_crop(img)
        img = self.add_gaussian_noise(img)
        img = self.random_blur(img)

        return img

class ColorPreservingTransforms:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg.INPUT.SIZE
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.aug_ratio = self.trainer.augmentation_params['augmentation_ratio']
        self.rotation = T.RandomRotation(
            degrees=self.trainer.augmentation_params['rotation_range']
        )
        self.crop = T.RandomResizedCrop(
            size=self.size,
            scale=self.trainer.augmentation_params['crop_scale']
        )
        self.gaussian_blur = T.GaussianBlur(
            kernel_size=3,
            sigma=self.trainer.augmentation_params['blur_radius']
        )
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

    def add_noise(self, img):
        """添加高斯噪声"""
        noise = torch.randn_like(img) * self.trainer.augmentation_params['noise_std']
        return torch.clamp(img + noise, 0, 1)

    def apply_random_augmentations(self, img):
        if self.trainer is None:
            raise ValueError("Trainer not set. Call set_trainer() first.")

        augmentations = []

        # 几何增强
        if torch.rand(1) < 0.7:
            augmentations.append(self.rotation)
        if torch.rand(1) < 0.7:
            augmentations.append(self.crop)

        # 高斯模糊
        if torch.rand(1) < self.trainer.augmentation_params['blur_prob']:
            augmentations.append(self.gaussian_blur)

        random.shuffle(augmentations)

        for aug in augmentations:
            img = aug(img)

        if torch.rand(1) < 0.5:
            img = self.add_noise(img)

        return img

    def __call__(self, img):
        if self.trainer is None:
            raise ValueError("Trainer not set. Call set_trainer() first.")

        if isinstance(img, torch.Tensor):
            if len(img.shape) == 4:
                batch_size = img.shape[0]
                augmented_images = []
                for i in range(batch_size):
                    for _ in range(self.aug_ratio):
                        aug_img = self.apply_random_augmentations(img[i].clone())
                        augmented_images.append(aug_img)
                return torch.stack(augmented_images)
            else:
                augmented_images = []
                for _ in range(self.aug_ratio):
                    aug_img = self.apply_random_augmentations(img.clone())
                    augmented_images.append(aug_img)
                return torch.stack(augmented_images)
        else:
            augmented_images = []
            for _ in range(self.aug_ratio):
                aug_img = self.apply_random_augmentations(img.copy())
                augmented_images.append(aug_img)
            return augmented_images

def get_train_transforms(cfg):
    return ColorPreservingTransforms(cfg)

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE),
        T.CenterCrop(cfg.INPUT.SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
