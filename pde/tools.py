from __future__ import annotations

from typing import Callable
from numpy.typing import ArrayLike


import numba as nb
import numpy as np

from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError

@nb.jit(nopython=True, cache=True)

def nb_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True)
    
    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0
    Returns
    -------
    2D array
        unique rows (or columns) from the input array
    1D array of ints
        indices of unique rows (or columns) in input array
    1D array of ints
        number of instances of each unique row
    """
    
    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()
        
    else:
        data = input_data.copy()
        
    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])
    
    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind = "mergesort")
        
        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]

    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1

    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx], orig_idx[idx], counts


@overload(np.all)
def np_all(x, axis = None):

    # ndarray.all with axis arguments for 2D arrays.

    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError("Only accepts NumPy ndarray")

    if not (1 <= x.ndim <= 2):
        raise TypingError("Only supports 1D or 2D NumPy ndarrays")

    if isinstance(axis, types.NoneType):

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 1:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 2:

        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)

        return _np_all_impl

    else:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl


def unique_rows(a: ArrayLike, **kwargs) -> np.ndarray | tuple[np.ndarray, ...]:
    # The numpy alternative `np.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    # a = np.asarray(a)

    a_shape = a.shape
    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))

    b = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    out = np.unique(b, **kwargs)
    # out[0] are the sorted, unique rows
    if isinstance(out, tuple):
        out = (out[0].view(a.dtype).reshape(out[0].shape[0], *a_shape[1:]), *out[1:])
    else:
        out = out.view(a.dtype).reshape(out.shape[0], *a_shape[1:])

    return out