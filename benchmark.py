from time import time
import numpy as np
import sys
import simd_example

# Use the first command-line argument to choose which
# function to call:
average_arrays = getattr(simd_example, sys.argv[1])


def timeit(title, arr1, arr2):
    start = time()
    for i in range(1_000):
        out = average_arrays(arr1, arr2)
    elapsed = (time() - start) / 1_000

    print(f"Time per call, {title}: {elapsed * 1000:.2} ms")


# Arrays laid out linearly in memory:
ARR1 = np.random.random((1_000_000,)).astype(np.float32)
ARR2 = np.random.random((1_000_000,)).astype(np.float32)
timeit("contiguous", ARR1, ARR2)

# Arrays where we grab every 16th item:
ARR3 = np.random.random((16_000_000,)).astype(np.float32)
ARR4 = np.random.random((16_000_000,)).astype(np.float32)
timeit("strided", ARR3[::16], ARR4[::16])
