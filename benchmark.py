from time import time
import numpy as np
from simd_example import average_arrays

# Two random arrays of unsigned integers
ARR1 = np.random.random((1_000_000,)).astype(np.float32)
ARR2 = np.random.random((1_000_000,)).astype(np.float32)

start = time()
for i in range(10_000):
    average_arrays(ARR1, ARR2)
elapsed = (time() - start) / 10_000

print("Time per call (milliseconds):", elapsed * 1000)
