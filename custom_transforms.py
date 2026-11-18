import random
import numpy as np
import torch
from PIL import Image

class InterpolationMode:
    BILINEAR = Image.BILINEAR
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS

class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size, size) if isinstance(size, int) else size
        self.interpolation = interpolation
    def __call__(self, img):
        return img.resize(self.size, self.interpolation)

class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size
    def __call__(self, img):
        th, tw = self.size
        w, h = img.size
        x1 = max(0, (w - tw) // 2)
        y1 = max(0, (h - th) // 2)
        return img.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size
    def __call__(self, img):
        th, tw = self.size
        w, h = img.size
        if w == tw and h == th: return img
        x1 = random.randint(0, max(0, w - tw))
        y1 = random.randint(0, max(0, h - th))
        return img.crop((x1, y1, x1 + tw, y1 + th))
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th: return 0, 0, h, w
        i = random.randint(0, max(0, h - th))
        j = random.randint(0, max(0, w - tw))
        return i, j, th, tw

class ToTensor:
    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img).float() / 255.0

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def crop(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))
