import torch
import torch.nn as nn
from torch.autograd import Variable

def get_in(x, k, df=None):
    """
    clojure's get_in function
    :param x:
    :param k:
    :return:
    """
    # fk = k.pop(0)
    # while fk and x:

    if isinstance(x, dict):
        if isinstance(k, str) and k in x:
            return x[k]
        elif isinstance(k, list):
            if len(k) == 0:
                return x
            elif len(k) == 1 and k[0] in x:
                return x[k[0]]
            elif k[0] in x:
                return get_in(x[k[0]], k[1:], df=df)
            else:
                return df
        else:
            return df
    elif isinstance(x, list):
        if isinstance(k, int) and len(x) > k:
            return x[k]
        elif isinstance(k, list):
            if len(k) == 0:
                return x
            elif len(k) == 1 and len(x) > k[0]:
                return x[k[0]]
            elif len(x) > k[0]:
                return get_in(x[k[0]], k[1:], df=df)
            else:
                return df
        else:
            return df
    else:
        return df


def identity(*x):
    return x


def _sumfn(x):
    pass


def summarize(model, fn=identity, o=0):
    res = []
    for key, module in model._modules.items():
        if type(module) in [nn.Container, nn.Sequential]:
            res += summarize(module, fn=fn, o=o + 3)
        else:
            summary = fn(key, module)
            if summary is not None:
                res.append(summary)
    return res


def _pr(x, d='size'):
    return '{} of {} : {} {}'.format(
        type(x).__name__, d, str(list(x.size())), str(x.storage_type()))


def _show(xs, ofs=0):
    st = '\n'
    if type(xs) in [list, tuple]:
        st += ' ' * ofs + str(type(xs)) + ' of len: ' + str(len(xs)) + ' containing:'
        nextl = [_show(x, ofs + 3) for x in xs]
        if len(set(nextl)) == 1:
            st += ' ' * ofs + str(nextl[0])
        else:
            for x in nextl:
                st += ' ' * ofs + x
    elif type(xs) in [nn.Container, nn.Sequential]:
        st += ' ' * ofs  + ' ModuleSeq containing:'
        st += _show(xs, ofs=ofs + 3)
    elif type(xs) == str:
        st += ' ' * ofs + xs
    elif torch.is_tensor(xs) is True:
        st += ' ' * ofs + _pr(xs, d='size')
    elif isinstance(xs, nn.Module):
        st += ' ' * ofs + ' Module containing : \n'
        for p in xs.parameters():
            st += _show(p, ofs=ofs+3)
        for key, module in xs._modules.items():
            st += _show(module, ofs=ofs+3)
    elif isinstance(xs, nn.Parameter):
        st += ' ' * ofs + '{} of size : {}'.format(type(xs).__name__,  str(list(xs.size())))
    elif isinstance(xs, Variable):
        st += ' ' * ofs + _pr(xs.data)
    ofs += 3
    return st


def show(*xs, pr=True ):
    res = _show(xs)
    if pr is True:
        print(res)
    return res

