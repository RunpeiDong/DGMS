from numpy.core.fromnumeric import argmax, size
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F

class RecognitionLosses(object):
    def __init__(self, size_average=True, batch_average=True, cuda=True, 
                     num_classes=10):
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.num_classes = num_classes

    def build_losses(self, mode='ce'):
        """Choices: ['ce' or 'focal' or 'mse']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        return nn.CrossEntropyLoss(logit, target)

    def FocalLoss(self, logit, target):
        return 
