import dataloader
import os
from torch.utils.data import dataset

import torchvision.transforms as transforms
import config as cfg

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

class ImageNet(Dataset):
    """`ImageNet <https://www.image-net.org/index.php>`_ Dataset.
    """
    def __init__(self, args, **kwargs):
        super(ImageNet, self).__init__()
        self.args = args
        self.num_class = cfg.NUM_CLASSES[args.dataset.lower()]

    @property
    def mean(self):
        return cfg.MEANS['imagenet']

    @property
    def std(self):
        return cfg.STDS['imagenet']

    def train_transform(self):
        return transforms.Compose([
            transforms.Resize(self.args.base_size),
            transforms.RandomCrop(self.args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.args.base_size),
            transforms.CenterCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def train_dataloader(self):
        dataset = ImageFolder(
            root=self.args.train_dir,
            transform=self.train_transform(),
        )
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        dataset = ImageFolder(
            root=self.args.val_dir,
            transform=self.val_transform(),
        )
        dataloader = DataLoader(dataset, batch_size=self.args.test_batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
