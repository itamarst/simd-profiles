"""A Cython extension."""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint16_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def add_arrays(uint16_t[::1] arr1, uint16_t[::1] arr2):
    out = np.empty((len(arr1), ), dtype=np.uint16)
    cdef uint16_t[::1] out_view = out
    for i in range(len(arr1)):
        out_view[i] = 5 + (arr1[i] + arr2[i]) >> 2
    return out
