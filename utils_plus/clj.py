import torch
from torch.nn import Parameter
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


def _show(xs, ofs=0):
    st = '\n'
    if type(xs) == list or type(xs) == tuple:
        st += ' ' * ofs + str(type(xs)) + ' of len: ' + str(len(xs)) + ' containing:'
        nextl = [_show(x, ofs + 3) for x in xs]
        if len(set(nextl)) == 1:
            st += ' ' * ofs + str(nextl[0])
        else:
            for x in nextl:
                st += ' ' * ofs + x
    elif type(xs) == str:
        st += ' ' * ofs + xs
    elif torch.is_tensor(xs) is True:
        st += ' ' * ofs + 'Tensor of size:   ' + str(list(xs.size())) \
              + ' ' + str(xs.storage_type())
    elif isinstance(xs, Parameter) is True:
        st += ' ' * ofs + 'Parameter of size:   ' + str(list(xs.size())) \
              + ' ' + str(xs.storage_type())
    elif isinstance(xs, Variable):
        st += ' ' * ofs + 'Variable of size: ' + str(list(xs.size())) \
              + ' ' + str(xs.data.storage_type())
    ofs += 3
    return st


def show(*xs, pr=True ):
    res = _show(xs)
    if pr is True:
        print(res)
    return res

