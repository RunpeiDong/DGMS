from .datasets import cifar10, imagenet, cub200, cars, aircraft

def make_data_loader(args, **kwargs):

    if args.dataset == 'cifar10':
        _cifar10 = cifar10.CIFAR10_Module(args, **kwargs)
        train_loader = _cifar10.train_dataloader()
        val_loader = _cifar10.val_dataloader()
        test_loader = None
        num_class = _cifar10.num_class

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cub200':
        _cub200 = cub200.CUB200(args, **kwargs)
        train_loader = _cub200.train_dataloader()
        val_loader = _cub200.val_dataloader()
        test_loader = None
        num_class = _cub200.num_class
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cars':
        _cars = cars.Cars(args, **kwargs)
        train_loader = _cars.train_dataloader()
        val_loader = _cars.val_dataloader()
        test_loader = None
        num_class = _cars.num_class
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'aircraft':
        _aircfraft = aircraft.Aircraft(args, **kwargs)
        train_loader = _aircfraft.train_dataloader()
        val_loader = _aircfraft.val_dataloader()
        test_loader = None
        num_class = _aircfraft.num_class
        
        return train_loader, val_loader, test_loader, num_class


    elif args.dataset == 'imagenet':
        _imagenet = imagenet.ImageNet(args, **kwargs)
        train_loader = _imagenet.train_dataloader()
        val_loader = _imagenet.val_dataloader()
        test_loader = None
        num_class = _imagenet.num_class

        return train_loader, val_loader, test_loader, num_class
