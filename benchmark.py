from time import time
import numpy as np
from simd_example import average_arrays


def timeit(title, arr1, arr2):
    start = time()
    for i in range(1_000):
        out = average_arrays(arr1, arr2)
    elapsed = (time() - start) / 1_000

    print(f"Time per call, {title}: {elapsed * 1000:.2} ms")


ARR = np.random.random((1_000_000)).astype(np.float32)
ARR2 = np.random.random((1_000_000)).astype(np.float32)
timeit("linear", ARR, ARR2)

ARR3 = np.random.random((16_000_000,)).astype(np.float32)
ARR4 = np.random.random((16_000_000,)).astype(np.float32)
timeit("strided", ARR3[::16], ARR4[::16])
