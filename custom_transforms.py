from PIL import Image
import torch
import numpy as np
import random

class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms: img = t(img)
        return img

class Resize:
    def __init__(self, size, interpolation=None):
        self.size = (size, size) if isinstance(size, int) else size
    def __call__(self, img): return img.resize(self.size, Image.BILINEAR)

class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size
    def __call__(self, img):
        w, h = img.size; th, tw = self.size
        return img.crop(((w - tw) // 2, (h - th) // 2, (w + tw) // 2, (h + th) // 2))

class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img):
        if random.random() < self.p: return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class ToTensor:
    def __call__(self, img):
        if img.mode != 'RGB': img = img.convert('RGB')
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, tensor): return (tensor - self.mean) / self.std

class InterpolationMode:
    BILINEAR = Image.BILINEAR
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST
