#ifndef IRIS_CUDA_H
#define IRIS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int iris_cuda_init(void);
int iris_cuda_available(void);
void iris_cuda_cleanup(void);
void iris_cuda_sync(void);

void iris_cuda_sgemm(int transpose_a, int transpose_b,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc);

void iris_cuda_sgemm_cached(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            float beta,
                            float *C, int ldc);

void iris_cuda_linear(float *y, const float *x, const float *W, const float *b,
                      int seq_len, int in_dim, int out_dim);

void iris_cuda_conv2d(float *out, const float *in, const float *weight, const float *bias,
                      int batch, int in_ch, int out_ch, int H, int W,
                      int kH, int kW, int stride, int padding);

int iris_cuda_attention(float *out,
                         const float *Q, const float *K, const float *V,
                         int seq_q, int seq_k, int heads, int head_dim,
                         float scale);

void iris_cuda_begin_batch(void);
void iris_cuda_end_batch(void);

/* Get device name for the banner. Returns static string, NULL if no GPU. */
const char *iris_cuda_device_name(void);

#ifdef __cplusplus
}
#endif

#endif
