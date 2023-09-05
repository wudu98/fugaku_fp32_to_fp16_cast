CXX = fcc
CXX_SRC = convert_benchmark.cpp
CFLAGS = -Nclang -Kfast,ocl,openmp -fPIC
BLAS = -SSL2BLAMP -lm -lfjomp

all: benchmark

benchmark:
	$(CXX) $(CXX_SRC) $(CFLAGS) $(BLAS) -o benchmark

