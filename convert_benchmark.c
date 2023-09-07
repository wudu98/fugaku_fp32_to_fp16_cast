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


void fp32_stream_copy_v0(int M, int N, int lda, int n_loops) {
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
  // printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  // printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((int*)A_out)[0]);
  // printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  // printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((int*)A_out)[1]);
  free(A_in);
  free(A_out);
}

void fp32_stream_copy_v1(int M, int N, int lda, int n_loops) {
  float *A_in  = malloc(M * lda * sizeof(float));
  float *A_out = malloc(M * lda * sizeof(float));
  init(A_in, M * lda);
  memset(A_out, 0.0, M * lda * sizeof(float));

  struct timespec start, end;
  double time_used = 0.0;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  for (int _loop = 0; _loop < n_loops; ++_loop) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++){
      for (int j = 0; j < N; j++){
        A_out[i * lda + j] = A_in[i * lda + j];
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  time_used = get_time(&start, &end);
  printf("fp32 stream copy latency: %.6f ms\n", time_used * 1e3 / n_loops);
  // printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  // printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((int*)A_out)[0]);
  // printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  // printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((int*)A_out)[1]);
  free(A_in);
  free(A_out);
}

void fp32_convert_fp16_copy_v0(int M, int N, int lda, int n_loops) {
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
  // printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  // printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((short*)A_out)[0]);
  // int tmp = *(int*)(&A_in[0]);
  // short t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[0] convert to fp16 = %#x\n", t);
  // printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  // printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((short*)A_out)[1]);
  // tmp = *(int*)(&A_in[1]);
  // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[1] convert to fp16 = %#x\n", t);
  // printf("A_in[2] = %.6f, A_out[2] = %.6f\n", A_in[2], A_out[2]);
  // printf("A_in[2] = %#x, A_out[2] = %#x\n", ((int*)A_in)[2], ((short*)A_out)[2]);
  // tmp = *(int*)(&A_in[2]);
  // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[2] convert to fp16 = %#x\n", t);
  free(A_in);
  free(A_out);
}

void fp32_convert_fp16_copy_v1(int M, int N, int lda, int n_loops) {
  float  *A_in  = malloc(M * lda * sizeof(float));
  __fp16 *A_out = malloc(M * lda * sizeof(__fp16));
  init(A_in, M * lda);
  memset(A_out, 0.0, M * lda * sizeof(__fp16));

  struct timespec start, end;
  double time_used = 0.0;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  for (int _loop = 0; _loop < n_loops; ++_loop) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++){
      for (int j = 0; j < N; j++){
        A_out[i * lda + j] = (__fp16)A_in[i * lda + j];
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  time_used = get_time(&start, &end);
  printf("fp32 convert fp16 latency: %.6f ms\n", time_used * 1e3 / n_loops);
  // printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  // printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((short*)A_out)[0]);
  // int tmp = *(int*)(&A_in[0]);
  // short t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[0] convert to fp16 = %#x\n", t);
  // printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  // printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((short*)A_out)[1]);
  // tmp = *(int*)(&A_in[1]);
  // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[1] convert to fp16 = %#x\n", t);
  // printf("A_in[2] = %.6f, A_out[2] = %.6f\n", A_in[2], A_out[2]);
  // printf("A_in[2] = %#x, A_out[2] = %#x\n", ((int*)A_in)[2], ((short*)A_out)[2]);
  // tmp = *(int*)(&A_in[2]);
  // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[2] convert to fp16 = %#x\n", t);
  free(A_in);
  free(A_out);
}


void fp32_convert_fp16_copy_v2(int M, int N, int lda, int n_loops) {
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
      for (int j = 0; j < N; j+=128){
        for (int jj = 0; jj < 128; jj++){
          A_out[i * lda + j + jj] = (__fp16)A_in[i * lda + j + jj];
        }
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  time_used = get_time(&start, &end);
  printf("fp32 convert fp16 latency: %.6f ms\n", time_used * 1e3 / n_loops);
  // printf("A_in[0] = %.6f, A_out[0] = %.6f\n", A_in[0], A_out[0]);
  // printf("A_in[0] = %#x, A_out[0] = %#x\n", ((int*)A_in)[0], ((short*)A_out)[0]);
  // int tmp = *(int*)(&A_in[0]);
  // short t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[0] convert to fp16 = %#x\n", t);
  // printf("A_in[1] = %.6f, A_out[1] = %.6f\n", A_in[1], A_out[1]);
  // printf("A_in[1] = %#x, A_out[1] = %#x\n", ((int*)A_in)[1], ((short*)A_out)[1]);
  // tmp = *(int*)(&A_in[1]);
  // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[1] convert to fp16 = %#x\n", t);
  // printf("A_in[2] = %.6f, A_out[2] = %.6f\n", A_in[2], A_out[2]);
  // printf("A_in[2] = %#x, A_out[2] = %#x\n", ((int*)A_in)[2], ((short*)A_out)[2]);
  // tmp = *(int*)(&A_in[2]);
  // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  // if (tmp & 0x1000) {t++;}
  // printf("double check A_in[2] convert to fp16 = %#x\n", t);
  free(A_in);
  free(A_out);
}


void fp32_convert_fp16_copy_v3(int M, int N, int lda, int n_loops) {
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
      for (int j = 0; j < N; j+=128){
        const offset = i * lda + j;
        asm volatile(
          "mov      x6, %[offset]                                  \n"
          "mov      x7, #64                                        \n"
          "ptrue    p0.b                                           \n"
          "add      x9,  %[A_in],  x6, lsl #2                      \n"
          "add      x10, x9,  x7                           \n"
          "add      x11, x10, x7                           \n"
          "add      x12, x11, x7                           \n"
          "add      x13, %[A_out], x6, lsl #1                      \n"
          "add      x14, x13, x7                           \n"
          "ld1w     z0.s, p0/z, [x9]                               \n"
          "ld1w     z1.s, p0/z, [x10]                              \n"
          "ld1w     z2.s, p0/z, [x11]                              \n"
          "ld1w     z3.s, p0/z, [x12]                              \n"

          "fcvt     z0.h, p0/m, z0.s                               \n"
          "fcvt     z1.h, p0/m, z1.s                               \n"
          "fcvt     z2.h, p0/m, z2.s                               \n"
          "fcvt     z3.h, p0/m, z3.s                               \n"

          "uzp1     z0.h, z0.h, z1.h                               \n"
          "uzp1     z2.h, z2.h, z3.h                               \n"

          "add      x9,  x12, x7                      \n"
          "add      x10, x9,  x7                           \n"
          "add      x11, x10, x7                           \n"
          "add      x12, x11, x7                           \n"

          "st1h     z0.h, p0,   [x13]                              \n"
          "st1h     z2.h, p0,   [x14]                              \n"

          "add      x13, x14, x7                      \n"
          "add      x14, x13, x7                           \n"

          "ld1w     z0.s, p0/z, [x9]                               \n"
          "ld1w     z1.s, p0/z, [x10]                              \n"
          "ld1w     z2.s, p0/z, [x11]                              \n"
          "ld1w     z3.s, p0/z, [x12]                              \n"

          "fcvt     z0.h, p0/m, z0.s                               \n"
          "fcvt     z1.h, p0/m, z1.s                               \n"
          "fcvt     z2.h, p0/m, z2.s                               \n"
          "fcvt     z3.h, p0/m, z3.s                               \n"

          "uzp1     z0.h, z0.h, z1.h                               \n"
          "uzp1     z2.h, z2.h, z3.h                               \n"
          "st1h     z0.h, p0,   [x13]                              \n"
          "st1h     z2.h, p0,   [x14]                              \n"

          : [A_out]"=r"(A_out)
          : "0"(A_out),
            [A_in]"r"(A_in),
            [offset]"r"(offset),
            [N]"r"(N)

          : "cc", "memory" , "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "z0", "z1", "z2", "z3"
        );
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
  printf("A_in[2] = %.6f, A_out[2] = %.6f\n", A_in[2], A_out[2]);
  printf("A_in[2] = %#x, A_out[2] = %#x\n", ((int*)A_in)[2], ((short*)A_out)[2]);
  tmp = *(int*)(&A_in[2]);
  t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
  if (tmp & 0x1000) {t++;}
  printf("double check A_in[2] convert to fp16 = %#x\n", t);

  int flag = 1;
  for (int i = 0; i < M; i++){
      for (int j = 0; j < N; j++){
        const offset = i * lda + j;

        if ( fabs((__fp16)A_in[offset] - A_out[offset] ) > 6e-3 && flag == 1){
          printf("error [%d][%d]\n", i, j);
          flag = 0;
        }
        // tmp = *(int*)(&A_in[offset]);
        // t = ((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) | (((tmp & 0x7f800000) >> 13) - ((127 - 15) << 10));
        // if (tmp & 0x1000) {t++;}
        // if ( *(short*)(&A_out[offset]) != t && flag == 1){
        //   printf("error [%d][%d], %#x, %#x\n", i, j, ((short*)A_out)[offset], t);
        //   flag = 0;
        // }
      }
  }
  if (flag == 1){
    printf("pass!\n");
  }

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

    fp32_stream_copy_v0(M, K, lda, 100);
    printf("-----------------------\n");
    
    fp32_stream_copy_v1(M, K, lda, 100);
    printf("-----------------------\n");

    fp32_convert_fp16_copy_v0(M, K, lda, 100);
    printf("-----------------------\n");

    fp32_convert_fp16_copy_v1(M, K, lda, 100);
    printf("-----------------------\n");

    fp32_convert_fp16_copy_v2(M, K, lda, 100);
    printf("-----------------------\n");

    fp32_convert_fp16_copy_v3(M, K, lda, 100);
    printf("-----------------------\n");
}