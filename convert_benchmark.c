#include <stdio.h>
#include <time.h>
#include <omp.h>

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

void* _mm_malloc(size_t align, size_t sz)
{
  void *ptr;
  int alloc_result = posix_memalign(&ptr, align, sz);
  if(alloc_result != 0)
  {
    return NULL;
  }
  return ptr;
}


void fp32_stream_copy(int M, int N, int lda, int n_loops) {
  // float *A_in  = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  // float *A_out = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  float *A_in  = malloc(M * lda * sizeof(float));
  float *A_out = malloc(M * lda * sizeof(float));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  for (int _loop = 0; _loop < n_loops; ++_loop) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++){
      for (int j = 0; j < N; j++){
        A_out[i * lda + j] = A_in[i * lda + j];
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double time_used = get_time(&start, &end);
  printf("fp32 stream copy latency: %.6f ms\n", time_used * 1e3/ n_loops);
  free(A_in);
  free(A_out);
}

void fp32_convert_fp16_copy(int M, int N, int lda, int n_loops) {
  // float *A_in  = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  // float *A_out = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  float *A_in  = malloc(M * lda * sizeof(float));
  __fp16 *A_out = malloc(M * lda * sizeof(__fp16));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  for (int _loop = 0; _loop < n_loops; ++_loop) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++){
      for (int j = 0; j < N; j++){
        A_out[i * lda + j] = (__fp16)A_in[i * lda + j];
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double time_used = get_time(&start, &end);
  printf("fp32 convert fp16 latency: %.6f ms\n", time_used * 1e3/ n_loops);
  free(A_in);
  free(A_out);
}

int main(){
    int M = 64;
    int N = 64;
    int lda = N;

    fp32_stream_copy(M, N, lda, 1000);
    fp32_convert_fp16_copy(M, N, lda, 1000);
}