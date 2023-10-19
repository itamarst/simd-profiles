"""A Cython extension."""

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def average_arrays(cnp.float32_t[::1] arr1, cnp.float32_t[::1] arr2):
    out = np.empty((len(arr1),), dtype=np.float32)
    cdef cnp.float32_t[::1] out_view = out
    for i in range(len(arr1)):
        out_view[i] = (arr1[i] + arr2[i]) / 2
    return out


def slow_average_arrays(cnp.float32_t[:] arr1, cnp.float32_t[:] arr2):
    out = np.empty(arr1.shape, dtype=np.float32)
    cdef cnp.float32_t[:] out_view = out.ravel()
    for i in range(len(arr1)):
        out_view[i] = (arr1[i] + arr2[i]) / 2
    return out
