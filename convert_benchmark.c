#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>
#include <cblas.h>

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void init(float *buf, int size) {
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
      buf[i] = 1.0f * rand() / RAND_MAX;
      //buf[i] = 1.0f;
  }
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
  float *A_in  = malloc(M * lda * sizeof(float));
  float *A_out = malloc(M * lda * sizeof(float));
  init(A_in, M * lda);
  memset(A_out, 0.0, M * lda * sizeof(float));

  struct timespec start, end;
  double time_used = 0.0;
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
  time_used = get_time(&start, &end);
  printf("fp32 stream copy latency: %.6f ms\n", time_used * 1e3 / n_loops);
  printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((int*)A_out)[0]);
  printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((int*)A_out)[1]);
  free(A_in);
  free(A_out);
}

void fp32_convert_fp16_copy(int M, int N, int lda, int n_loops) {
  float  *A_in  = malloc(M * lda * sizeof(float));
  __fp16 *A_out = malloc(M * lda * sizeof(__fp16));
  init(A_in, M * lda);
  memset(A_out, 0.0, M * lda * sizeof(__fp16));

  struct timespec start, end;
  double time_used = 0.0;
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
  time_used = get_time(&start, &end);
  printf("fp32 convert fp16 latency: %.6f ms\n", time_used * 1e3 / n_loops);
  printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((short*)A_out)[0]);
  int tmp = *(int*)(&A_in[0]);
  short t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  if (tmp & 0x1000) {t++;}
  printf("double check A_in[0] convert to fp16 = %#x\n", t);
  printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((short*)A_out)[1]);
  tmp = *(int*)(&A_in[1]);
  t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  if (tmp & 0x1000) {t++;}
  printf("double check A_in[1] convert to fp16 = %#x\n", t);
  free(A_in);
  free(A_out);
}

int main(){
    int num_threads = 48;
    omp_set_num_threads(num_threads);
    printf("number of threads = %d\n", omp_get_max_threads());
    printf("-----------------------\n");

    int M = 256;
    int K = 256;
    int lda = K;

    fp32_stream_copy(M, K, lda, 10);
    printf("-----------------------\n");

    fp32_convert_fp16_copy(M, K, lda, 10);
    printf("-----------------------\n");
}