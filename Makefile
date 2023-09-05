CXX = fcc
CXX_SRC = convert_benchmark.c
CFLAGS = -std=gnu11 -Nclang -Kfast,ocl,openmp -fPIC
BLAS = -SSL2BLAMP -lm -lfjomp

all: benchmark

benchmark:
	$(CXX) $(CXX_SRC) $(CFLAGS) $(BLAS) -o benchmark

