import torch

def proxyless_nas_mobile(args):
    target_platform = "proxyless_mobile" # proxyless_gpu, proxyless_mobile, proxyless_mobile14 are also avaliable.
    if args.pretrained:
        model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
        print("ImageNet pretrained ProxylessNAS-Mobile loaded! (Pretrained Top-1 Acc: 74.59%)")
    else:
        model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
    return model
