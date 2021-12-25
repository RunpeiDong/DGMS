import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

def resnet(args, **kwargs):
    """Constructs a ResNet model."""
    if args.pretrained:
        model = models.__dict__[args.network](pretrained=True)
        print("ImageNet pretrained model loaded!")
    else:
        model = models.__dict__[args.network]()
    return model
