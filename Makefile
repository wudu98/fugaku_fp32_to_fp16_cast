CXX = fcc
CFLAGS = -std=gnu11 -Nclang -Kfast,ocl,openmp -fPIC
BLAS = -SSL2BLAMP -lm -lfjomp

CXX_SRC = convert_benchmark.c

all: benchmark

benchmark:
	$(CXX) $(CXX_SRC) $(CFLAGS) $(BLAS) -o benchmark

clean:
	rm -rf benchmark

