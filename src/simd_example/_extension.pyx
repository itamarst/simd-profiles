"""A Cython extension."""

import numpy as np
cimport numpy as cnp
cimport cython


def average_arrays_1(cnp.float32_t[:] arr1, cnp.float32_t[:] arr2):
    out = np.empty((len(arr1), ), dtype=np.float32)
    cdef cnp.float32_t[:] out_view = out
    for i in range(len(arr1)):
        out_view[i] = (arr1[i] + arr2[i]) / 2
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def average_arrays_2(cnp.float32_t[:] arr1, cnp.float32_t[:] arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be the same size")
    out = np.empty((len(arr1), ), dtype=np.float32)
    cdef cnp.float32_t[:] out_view = out
    for i in range(len(arr1)):
        out_view[i] = (arr1[i] + arr2[i]) / 2
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def average_arrays_3(cnp.float32_t[::1] arr1, cnp.float32_t[::1] arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be the same size")
    out = np.empty((len(arr1), ), dtype=np.float32)
    cdef cnp.float32_t[::1] out_view = out
    for i in range(len(arr1)):
        out_view[i] = (arr1[i] + arr2[i]) / 2
    return out


# This type will be one of these two underlying types at compile time:
ctypedef fused float_arr_t:
   cnp.float32_t[:]
   cnp.float32_t[::1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef average_arrays_impl(float_arr_t arr1, float_arr_t arr2):
    out = np.empty((len(arr1), ), dtype=np.float32)
    cdef cnp.float32_t[::1] out_view = out
    for i in range(len(arr1)):
        out_view[i] = (arr1[i] + arr2[i]) / 2
    return out

def average_arrays_4(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be the same size")
    if arr1.data.contiguous and arr2.data.contiguous:
        return average_arrays_impl[cnp.float32_t[::1]](arr1, arr2)
    else:
        return average_arrays_impl[cnp.float32_t[:]](arr1, arr2)
