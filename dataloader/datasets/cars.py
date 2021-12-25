import torch
import torchvision
import config as cfg

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

class Cars(Dataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    """
    def __init__(self, args, **kwargs):
        super(Cars, self).__init__()
        self.args = args
        self.num_class = cfg.NUM_CLASSES[args.dataset.lower()]

    @property
    def mean(self):
        return cfg.MEANS[self.args.dataset.lower()]

    @property
    def std(self):
        return cfg.STDS[self.args.dataset.lower()]

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
 