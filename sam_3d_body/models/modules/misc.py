# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections.abc
from itertools import repeat


# From PyTorch internals
def _ntuple(n):
    """A `to_tuple` function generator.

    It returns a function, this function will repeat the input to a tuple of
    length ``n`` if the input is not an Iterable object, otherwise, return the
    input directly.

    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
