import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class VGG_SMALL(nn.Module):
    """ Pytorch implementation of VGGSmall artecture, modified from 
    https://github.com/microsoft/LQ-Nets/blob/master/cifar10-vgg-small.py (Tensorflow Version).
    """
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG_SMALL, self).__init__()
        self.features = features
        self.classifier = nn.Linear(in_features=512*4*4, out_features=num_classes, bias=True)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    batch_norm = True
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d]
            batch_norm = not batch_norm
            in_channels = v
    return nn.Sequential(*layers)

def vggsmall(args):
    config = [128, 128, 'M', 'A', 256, 256, 'M', 'A', 512, 512, 'M', 'A']
    return VGG_SMALL(make_layers(config))
    