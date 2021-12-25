import math
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, ReduceLROnPlateau

def get_scheduler(args, optimizer, base_lr, steps_per_epoch=0):
    mode = args.lr_scheduler
    print('Using {} LR Scheduler!'.format(mode))
    if mode == 'one-cycle':
        scheduler = OneCycleLR(optimizer, base_lr,
                               steps_per_epoch=steps_per_epoch,
                               epochs=args.epochs)
    elif mode == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch)
    elif mode == 'multi-step':
        scheduler = MultiStepLR(optimizer, milestones=[
                                e * steps_per_epoch for e in args.schedule], gamma=0.1)
    else:
        assert mode == 'reduce'
        scheduler = ReduceLROnPlateau(optimizer)

    return scheduler
