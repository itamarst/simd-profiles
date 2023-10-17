default:
	env CFLAGS="-O3" pip install --quiet .
	python benchmark.py

native:
	env CFLAGS="-O3 -march=native" pip install --quiet .
	python benchmark.py

avx2:
	env CFLAGS="-O3 -march=x86-64-v3" pip install --quiet .
	python benchmark.py
