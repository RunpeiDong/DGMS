from torch.utils import data
import config as cfg
from config import DATA_FOLDERS
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cifar10':
            return cfg.DATA_FOLDERS['cifar']
        elif dataset == 'imagenet':
            return cfg.DATA_FOLDERS['imagenet']
        elif dataset == 'cub200':
            return cfg.DATA_FOLDERS['cub200']
        elif dataset == 'cars':
            return cfg.DATA_FOLDERS['cars']
        elif dataset == 'aircraft':
            return cfg.DATA_FOLDERS['aircraft']
        else:
            raise NotImplementedError("no support for dataset " + dataset)