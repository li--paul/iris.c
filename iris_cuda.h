#ifndef IRIS_CUDA_H
#define IRIS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iris_cuda_tensor *iris_cuda_tensor_t;

int iris_cuda_init(void);
int iris_cuda_available(void);
void iris_cuda_cleanup(void);
void iris_cuda_sync(void);

iris_cuda_tensor_t iris_cuda_tensor_create(const float *data, size_t num_elements);
iris_cuda_tensor_t iris_cuda_tensor_alloc(size_t num_elements);
void iris_cuda_tensor_free(iris_cuda_tensor_t tensor);
int iris_cuda_tensor_read(iris_cuda_tensor_t tensor, float *out);
int iris_cuda_tensor_write(iris_cuda_tensor_t tensor, const float *data);
float *iris_cuda_tensor_data(iris_cuda_tensor_t tensor);
size_t iris_cuda_tensor_size(iris_cuda_tensor_t tensor);

int iris_cuda_linear_tensor(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                            const float *W, int seq_len, int in_dim, int out_dim,
                            int cache_weight);
int iris_cuda_split_qkv_mlp(iris_cuda_tensor_t fused,
                            iris_cuda_tensor_t q, iris_cuda_tensor_t k, iris_cuda_tensor_t v,
                            iris_cuda_tensor_t gate, iris_cuda_tensor_t up,
                            int seq, int hidden, int mlp_hidden);
int iris_cuda_qk_rms_norm(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps);
int iris_cuda_rope_unified(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                           const float *txt_cos, const float *txt_sin,
                           const float *img_cos, const float *img_sin,
                           int seq, int img_offset, int heads, int head_dim);
int iris_cuda_attention_tensor(iris_cuda_tensor_t out,
                               iris_cuda_tensor_t q, iris_cuda_tensor_t k, iris_cuda_tensor_t v,
                               int seq_q, int seq_k, int heads, int head_dim, float scale);
int iris_cuda_silu_mul_tensor(iris_cuda_tensor_t gate, iris_cuda_tensor_t up, int n);
int iris_cuda_concat_attn_mlp(iris_cuda_tensor_t attn, iris_cuda_tensor_t mlp,
                              iris_cuda_tensor_t out, int seq, int hidden, int mlp_hidden);
int iris_cuda_gated_add(iris_cuda_tensor_t hidden, const float *gate,
                        iris_cuda_tensor_t proj, int seq, int hidden_dim);

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
int iris_cuda_split_qkv_mlp_device(const float *d_fused,
                                   float *d_q, float *d_k, float *d_v,
                                   float *d_gate, float *d_up,
                                   int seq, int hidden, int mlp_hidden);
int iris_cuda_qk_rms_norm_device(float *d_q, float *d_k,
                                 const float *d_q_weight, const float *d_k_weight,
                                 int seq, int heads, int head_dim, float eps);
int iris_cuda_rope_unified_device(float *d_q, float *d_k,
                                  const float *d_txt_cos, const float *d_txt_sin,
                                  const float *d_img_cos, const float *d_img_sin,
                                  int seq, int img_offset, int heads, int head_dim);
int iris_cuda_concat_attn_mlp_device(const float *d_attn, const float *d_mlp,
                                     float *d_out, int seq, int hidden, int mlp_hidden);
int iris_cuda_gated_add_device(float *d_hidden, const float *d_gate,
                               const float *d_proj, int seq, int hidden);
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
