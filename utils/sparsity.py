from modeling.DGMS import DGMSConv
import torch
import torch.nn as nn

def _count_zero(x):
    return x.eq(0.0).float().mean().item()


def _check_filter(x):
    return _count_zero(x.abs().sum(dim=(1, 2, 3)))


def _check_channel(x):
    return _count_zero(x.abs().sum(dim=(0, 2, 3)))


def _check_kernel(x):
    return _count_zero(x.abs().sum(dim=(2, 3)))


def _check_irregular(x):
    return _count_zero(x)

def check_total_zero(x):
    with torch.no_grad():
        return x.eq(0.0).float().sum().item()

def check_total_weights(x):
    with torch.no_grad():
        return x.numel()


_CHECKS = {
    'filter': _check_filter,
    'kernel': _check_kernel,
    'channel': _check_channel,
    'irregular': _check_irregular,
}


def check(x, method):
    with torch.no_grad():
        return _CHECKS[method](x)


class SparsityMeasure(object):
    def __init__(self, args):
        super(SparsityMeasure, self).__init__()
        self.args = args

    def check_sparsity_per_layer(self, model):
        total_sparsity_num = 0
        total_weight_num = 0
        skipped_weight_num = 0
        for name, m in model.named_modules():
            if isinstance(m, DGMSConv):
                Pweight = m.get_Pweight()
                sparse_ratio = check(Pweight, 'irregular')
                total_sparsity_num += check_total_zero(Pweight)
                total_weight_num += check_total_weights(Pweight)
                print(f'{name}\t{m.weight.size()}:\t{sparse_ratio:.3f}')
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                skipped_weight_num += check_total_weights(m.weight)
                print(f'{name}\t{m.weight.size()}:\tfp32')
        total_sparse_ratio = total_sparsity_num / total_weight_num
        nz_parameters_num = total_weight_num-total_sparsity_num
        print(f"Total sparsity is {total_sparsity_num} / {total_weight_num}:\t {total_sparse_ratio:.4f}")
        nz_ratio = 1 - total_sparse_ratio
        print(f"NZ ratio is :\t {nz_ratio:.4f}")
        model_params = (skipped_weight_num+nz_parameters_num) / 1e6
        print(f"Skipped weights number: {skipped_weight_num}")
        print(f"NZ parameters size: {model_params:.2f}M")
        return total_sparse_ratio, model_params
