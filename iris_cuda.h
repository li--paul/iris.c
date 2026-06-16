/*
 * Iris CUDA Acceleration
 *
 * GPU-accelerated tensor operations using NVIDIA CUDA and cuBLAS.
 * Mirrors the Metal tensor API (iris_gpu_tensor_t) for the CUDA backend.
 *
 * Design:
 *   - f32 tensors backed by cudaMalloc device memory
 *   - Buffer pool for scratch tensors (avoids per-op cudaMalloc/free)
 *   - Weight cache: upload f32 weights once, cache by CPU pointer
 *   - All operations are async on the default stream
 *   - iris_cuda_sync() / batch mode for explicit sync control
 *   - Single-block chained path: upload hidden once, run all blocks, download once
 */

#ifndef IRIS_CUDA_H
#define IRIS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

/* Initialize CUDA. Returns 1 on success, 0 if no CUDA device available. */
int iris_cuda_init(void);

/* Check if CUDA is available and initialized. */
int iris_cuda_available(void);

/* Cleanup all CUDA resources (buffers, caches, pools). */
void iris_cuda_cleanup(void);

/* Reset transient state (activation pool) while preserving weight cache. */
void iris_cuda_reset_transient(void);

/* ========================================================================
 * Sync & Batch
 * ======================================================================== */

/* Synchronize: wait for all pending GPU work to complete. */
void iris_cuda_sync(void);

/* Begin batch mode: operations are queued without sync until end_batch. */
void iris_cuda_begin_batch(void);

/* End batch mode: sync all queued operations. */
void iris_cuda_end_batch(void);

/* Check if currently in batch mode. */
int iris_cuda_in_batch(void);

/* ========================================================================
 * CUDA Tensor API
 * ======================================================================== */

typedef struct iris_cuda_tensor *iris_cuda_tensor_t;

/* Create a GPU tensor from CPU data (cudaMemcpy upload). */
iris_cuda_tensor_t iris_cuda_tensor_create(const float *data, size_t num_elements);

/* Allocate an uninitialized GPU tensor (for output buffers). */
iris_cuda_tensor_t iris_cuda_tensor_alloc(size_t num_elements);

/* Allocate a persistent GPU tensor (not returned to pool on free). */
iris_cuda_tensor_t iris_cuda_tensor_alloc_persistent(size_t num_elements);

/* Read tensor data back to CPU (cudaMemcpy download). Syncs if work is pending. */
void iris_cuda_tensor_read(iris_cuda_tensor_t tensor, float *out);

/* Write CPU data to tensor (cudaMemcpy upload). Syncs if work is pending. */
void iris_cuda_tensor_write(iris_cuda_tensor_t tensor, const float *data);

/* Release tensor back to pool (or free if persistent). */
void iris_cuda_tensor_free(iris_cuda_tensor_t tensor);

/* Get raw device pointer (for use with cuBLAS custom kernels). */
float *iris_cuda_tensor_data(iris_cuda_tensor_t tensor);

/* Number of elements in the tensor. */
size_t iris_cuda_tensor_size(iris_cuda_tensor_t tensor);

/* ========================================================================
 * CUDA Operations on Device Tensors (f32, all device-in device-out)
 * ======================================================================== */

/*
 * Linear layer on GPU: out = x @ W^T + b
 * x:  [seq, in_dim] device tensor
 * W:  [out_dim, in_dim] CPU pointer (cached on GPU after first call)
 * b:  [out_dim] CPU pointer (can be NULL)
 * out: [seq, out_dim] device tensor (pre-allocated via iris_cuda_tensor_alloc)
 * Returns 1 on success, 0 on failure.
 */
int iris_cuda_linear_into(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                          const float *W, const float *b,
                          int seq_len, int in_dim, int out_dim);

/*
 * Linear layer with bf16 weights: out = x @ W_bf16^T
 * W_bf16: [out_dim, in_dim] bf16 CPU pointer (cached as f32 on GPU)
 */
int iris_cuda_linear_bf16_into(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                               const uint16_t *W_bf16,
                               int seq_len, int in_dim, int out_dim);

/*
 * AdaLN (Adaptive Layer Norm): out = (1 + scale) * rmsnorm(x) + shift
 * All tensors are device-resident [seq, hidden].
 */
int iris_cuda_adaln_norm(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                         const float *shift, const float *scale,
                         int seq, int hidden, float eps);

/*
 * QK RMSNorm: separate RMSNorm on q and k with learned weights.
 * q/k: [seq, heads * head_dim] device tensors, modified in-place.
 */
int iris_cuda_qk_rms_norm(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                          const float *weight_q, const float *weight_k,
                          int seq, int heads, int head_dim, float eps);

/*
 * Split fused QKV+MLP into q, k, v, gate, up.
 * fused: [seq, hidden*3 + mlp_hidden*2]
 * q/k/v: [seq, hidden], gate/up: [seq, mlp_hidden]
 */
int iris_cuda_split_qkv_mlp(iris_cuda_tensor_t fused,
                            iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                            iris_cuda_tensor_t v,
                            iris_cuda_tensor_t gate, iris_cuda_tensor_t up,
                            int seq, int hidden, int mlp_hidden);

/*
 * Apply 2D rotary position embeddings (Flux style).
 * q/k: [seq, heads * head_dim] modified in-place.
 * cos/sin: [seq, head_dim / 2] pre-computed frequencies.
 */
int iris_cuda_rope_2d(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                      const float *cos_freq, const float *sin_freq,
                      int seq, int heads, int head_dim);

/*
 * Scaled dot-product attention (cuBLAS-based).
 * Q/K/V: [seq, heads * head_dim], out: [seq, heads * head_dim]
 */
int iris_cuda_attention(iris_cuda_tensor_t out,
                        iris_cuda_tensor_t Q, iris_cuda_tensor_t K,
                        iris_cuda_tensor_t V,
                        int seq, int heads, int head_dim, float scale);

/*
 * SiLU(gate) * up element-wise (SwiGLU activation).
 * gate is modified in-place; up is read-only.
 */
int iris_cuda_silu_mul(iris_cuda_tensor_t gate, iris_cuda_tensor_t up, int n);

/*
 * Concat attn_out and mlp_out along last dim.
 * attn: [seq, hidden], mlp: [seq, mlp_hidden], out: [seq, hidden + mlp_hidden]
 */
int iris_cuda_concat_attn_mlp(iris_cuda_tensor_t attn, iris_cuda_tensor_t mlp,
                              iris_cuda_tensor_t out,
                              int seq, int hidden, int mlp_hidden);

/*
 * Gated residual add: out += gate * proj_out
 * out: [seq, hidden] modified in-place.
 */
int iris_cuda_gated_add(iris_cuda_tensor_t out, iris_cuda_tensor_t proj_out,
                        const float *gate, int seq, int hidden);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_CUDA_H */
