from time import time
import numpy as np
from simd_example import average_arrays

# Two random arrays of unsigned integers
ARR1 = np.random.random((1_000_000,)).astype(np.float32)
ARR2 = np.random.random((1_000_000,)).astype(np.float32)

start = time()
for i in range(1_000):
    average_arrays(ARR1, ARR2)
elapsed = (time() - start) / 1_000

print(f"Time per call: {elapsed * 1000:.2} ms")
