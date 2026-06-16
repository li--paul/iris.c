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

int iris_cuda_rms_norm(float *out, const float *x, const float *weight,
                       int seq_len, int hidden, float eps);

int iris_cuda_adaln_norm(float *out, const float *x,
                         const float *shift, const float *scale,
                         int seq_len, int hidden, float eps);

int iris_cuda_silu_mul(float *gate, const float *up, int n);

int iris_cuda_apply_rope(float *x, const float *freqs,
                         int batch, int seq, int heads, int head_dim);

void iris_cuda_conv2d(float *out, const float *in, const float *weight, const float *bias,
                      int batch, int in_ch, int out_ch, int H, int W,
                      int kH, int kW, int stride, int padding);

int iris_cuda_attention(float *out,
                         const float *Q, const float *K, const float *V,
                         int seq_q, int seq_k, int heads, int head_dim,
                         float scale);

int iris_cuda_softmax_inplace(float *d_x, int rows, int cols);
int iris_cuda_transpose_shd_to_hsd(const float *d_src, float *d_dst,
                                   int seq, int heads, int head_dim);
int iris_cuda_transpose_hsd_to_shd(const float *d_src, float *d_dst,
                                   int seq, int heads, int head_dim);
int iris_cuda_add_bias_rows(float *d_x, const float *d_bias, int rows, int cols);
int iris_cuda_im2col(const float *d_in, float *d_col,
                     int in_ch, int H, int W, int kH, int kW,
                     int stride, int padding, int outH, int outW);
int iris_cuda_conv2d_add_bias(float *d_out, const float *d_bias,
                              int batch, int out_ch, int pixels);
int iris_cuda_attention_fused_device(float *d_out,
                                     const float *d_Q, const float *d_K, const float *d_V,
                                     int seq_q, int seq_k, int heads, int head_dim,
                                     float scale);
int iris_cuda_adaln_norm_device(float *d_out, const float *d_x,
                                const float *d_shift, const float *d_scale,
                                int rows, int hidden, float eps);
int iris_cuda_apply_rope_device(float *d_x, const float *d_freqs,
                                int batch, int seq, int heads, int head_dim);

void iris_cuda_begin_batch(void);
void iris_cuda_end_batch(void);

/* Get device name for the banner. Returns static string, NULL if no GPU. */
const char *iris_cuda_device_name(void);

#ifdef __cplusplus
}
#endif

#endif
