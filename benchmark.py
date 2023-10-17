import numpy as np
from time import time

# Two random arrays of unsigned integers
ARR1 = np.random.randint(0, 1_000, (1_000_000), dtype=np.uint16)
ARR2 = np.random.randint(0, 1_000, (1_000_000), dtype=np.uint16)

from simd_example import add_arrays

start = time()
for i in range(10_000):
    add_arrays(ARR1, ARR2)
elapsed = (time() - start) / 10_000

print("Time per call (milliseconds):", elapsed * 1000)
