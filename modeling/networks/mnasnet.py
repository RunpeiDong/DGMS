import torch
import torch.nn as nn
import torchvision.models as models


def mnasnet0_5(args, **kwargs):
    if args.pretrained:
        model = models.__dict__['mnasnet0_5'](pretrained=True)
        print("ImageNet pretrained model loaded!")
    else:
         model = models.__dict__['mnasnet0_5']()
    return model

def mnasnet1_0(args, **kwargs):
    if args.pretrained:
        model = models.mnasnet1_0(pretrained=True)
        print("ImageNet pretrained model loaded!")
    else:
         model = models.mnasnet1_0()
    return model
