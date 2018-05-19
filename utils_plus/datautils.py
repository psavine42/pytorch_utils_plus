import torch
from torch.autograd import Variable


def cuda_if_available(xs, is_cuda=False, **kwargs):
    if is_cuda is True:
        return Variable(xs, **kwargs).cuda()
    else:
        return Variable(xs, **kwargs)


def package(xs, is_cuda=False, **kwargs):
    """Variable of correct type around tensor
        or tuple of tensors."""
    if torch.is_tensor(xs) is True:
        return cuda_if_available(xs, is_cuda=is_cuda, **kwargs)
    elif xs is None:
        return None
    elif type(xs) == Variable:
        return xs
    elif type(xs) == tuple:
        return tuple([package(v, is_cuda=is_cuda, **kwargs) for v in xs])
    elif type(xs) == list:
        return [package(v, is_cuda=is_cuda, **kwargs) for v in xs]


def repackage(xs, **kwargs):
    """Wraps hidden states in new Variables,
            to detach them from their history."""
    if type(xs) == Variable:
        return cuda_if_available(xs.data, **kwargs)
    elif torch.is_tensor(xs) is True:
        return cuda_if_available(xs, **kwargs)
    elif xs is None:
            return None
    elif type(xs) == list:
        return [repackage(v, **kwargs) for v in xs]
    else:
        return tuple(repackage(v, **kwargs) for v in xs)


def depckg(xs):
    if type(xs) == Variable:
        return depckg(xs.data)
    elif torch.is_tensor(xs) is True:
        return xs.cpu()
    elif xs is None:
        return None
    else:
        return tuple(depckg(v) for v in xs)


