import torchvision.transforms as transforms
import config as cfg

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

class CIFAR10_Module(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    NUM_CLASSES = cfg.NUM_CLASSES['cifar10']

    def __init__(self, args, **kwargs):
        super(CIFAR10_Module, self).__init__()
        self.args = args

    @property
    def mean(self):
        return cfg.MEANS['cifar']

    @property
    def std(self):
        return cfg.STDS['cifar']

    @property
    def num_class(self):
        return self.NUM_CLASSES

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.args.train_dir,
                          train=True, download=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader

    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.args.val_dir,
                          train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, 
                                num_workers=4, pin_memory=True)
        return dataloader

class CIFAR100_Module(CIFAR10_Module):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    
    def __init__(self, args, **kwargs):
        super(CIFAR100_Module, self).__init__(args, **kwargs)

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR100(root=self.args.train_dir,
                          train=True, download=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.args.test_batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader

    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR100(root=self.args.val_dir,
                          train=False, transform=transform_val)
        dataloader = DataLoader(
            dataset, batch_size=self.args.batch_size, num_workers=4, pin_memory=True)
        return dataloader
