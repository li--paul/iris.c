/*
 * Iris CUDA Acceleration — Implementation
 *
 * GPU-accelerated tensor operations using NVIDIA CUDA and cuBLAS.
 * Provides device-resident tensor API for chained single-block inference.
 */

#include "iris_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ========================================================================
 * CUDA Tensor Struct
 * ======================================================================== */

struct iris_cuda_tensor {
    float *device_ptr;
    size_t num_elements;
    int has_pending_work;
    int persistent;
};

/* ========================================================================
 * Global State
 * ======================================================================== */

static int g_initialized = 0;
static cublasHandle_t g_cublas = NULL;
static cudaStream_t g_stream = NULL;

/* Batch mode: queue ops without sync until end_batch */
static int g_batch_mode = 0;

/* Pool for scratch tensors — pre-allocated chunks reused via free-list */
#define MAX_POOL_CHUNKS 512
#define POOL_CHUNK_SIZE (8 * 1024 * 1024)  /* 8 MB per chunk (in floats) */

typedef struct pool_chunk {
    float *device_ptr;
    size_t capacity;  /* in floats */
    int in_use;
    struct pool_chunk *next;
} pool_chunk_t;

static pool_chunk_t *g_pool_head = NULL;
static int g_pool_chunks_used = 0;

/* Weight cache: maps CPU pointer -> GPU device pointer (f32, uploaded once) */
#define MAX_WEIGHT_ENTRIES 512

typedef struct {
    const void *cpu_ptr;
    float *gpu_ptr;
    size_t num_elements;
    int is_bf16;  /* 1 if source was bf16, converted to f32 on upload */
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[MAX_WEIGHT_ENTRIES];
static int g_weight_cache_count = 0;

/* ========================================================================
 * Internal Helpers
 * ======================================================================== */

static void cuda_check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(err));
    }
}

static int cublas_check(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error [%s]: %d\n", msg, (int)status);
        return 0;
    }
    return 1;
}

/* ========================================================================
 * Buffer Pool
 * ======================================================================== */

static float *pool_alloc(size_t num_elements) {
    pool_chunk_t *chunk = g_pool_head;
    while (chunk) {
        if (!chunk->in_use && chunk->capacity >= num_elements) {
            chunk->in_use = 1;
            return chunk->device_ptr;
        }
        chunk = chunk->next;
    }

    if (g_pool_chunks_used >= MAX_POOL_CHUNKS) return NULL;

    size_t alloc_size = num_elements > POOL_CHUNK_SIZE ? num_elements : POOL_CHUNK_SIZE;
    float *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, alloc_size * sizeof(float));
    if (err != cudaSuccess) return NULL;

    pool_chunk_t *new_chunk = (pool_chunk_t *)malloc(sizeof(pool_chunk_t));
    if (!new_chunk) { cudaFree(ptr); return NULL; }
    new_chunk->device_ptr = ptr;
    new_chunk->capacity = alloc_size;
    new_chunk->in_use = 1;
    new_chunk->next = g_pool_head;
    g_pool_head = new_chunk;
    g_pool_chunks_used++;
    return ptr;
}

static void pool_release(float *device_ptr) {
    pool_chunk_t *chunk = g_pool_head;
    while (chunk) {
        if (chunk->device_ptr == device_ptr) {
            chunk->in_use = 0;
            return;
        }
        chunk = chunk->next;
    }
}

static void pool_destroy(void) {
    pool_chunk_t *chunk = g_pool_head;
    while (chunk) {
        pool_chunk_t *next = chunk->next;
        cudaFree(chunk->device_ptr);
        free(chunk);
        chunk = next;
    }
    g_pool_head = NULL;
    g_pool_chunks_used = 0;
}

static void pool_reset_transient(void) {
    pool_chunk_t *chunk = g_pool_head;
    while (chunk) {
        chunk->in_use = 0;
        chunk = chunk->next;
    }
}

/* ========================================================================
 * Weight Cache
 * ======================================================================== */

/* Forward declarations of CUDA kernels defined below */
__global__ void bf16_to_f32_kernel(float *out, const uint16_t *in, int n);

static float *weight_cache_get(const void *cpu_ptr, size_t num_elements, int is_bf16) {
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == cpu_ptr && g_weight_cache[i].is_bf16 == is_bf16) {
            return g_weight_cache[i].gpu_ptr;
        }
    }

    if (g_weight_cache_count >= MAX_WEIGHT_ENTRIES) return NULL;

    float *gpu_ptr = NULL;
    cudaError_t ce = cudaMalloc(&gpu_ptr, num_elements * sizeof(float));
    if (ce != cudaSuccess || !gpu_ptr) return NULL;
    gpu_ptr = gpu_ptr; /* suppress unused warning */

    if (is_bf16) {
        uint16_t *tmp = NULL;
        ce = cudaMalloc(&tmp, num_elements * sizeof(uint16_t));
        if (ce != cudaSuccess || !tmp) { cudaFree(gpu_ptr); return NULL; }
        ce = cudaMemcpy(tmp, cpu_ptr, num_elements * sizeof(uint16_t), cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) { cudaFree(tmp); cudaFree(gpu_ptr); return NULL; }
        dim3 block(256);
        dim3 grid((num_elements + 255) / 256);
        bf16_to_f32_kernel<<<grid, block, 0, g_stream>>>(gpu_ptr, tmp, (int)num_elements);
        cudaFree(tmp);
    } else {
        ce = cudaMemcpy(gpu_ptr, cpu_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) { cudaFree(gpu_ptr); return NULL; }
    }

    g_weight_cache[g_weight_cache_count].cpu_ptr = cpu_ptr;
    g_weight_cache[g_weight_cache_count].gpu_ptr = gpu_ptr;
    g_weight_cache[g_weight_cache_count].num_elements = num_elements;
    g_weight_cache[g_weight_cache_count].is_bf16 = is_bf16;
    g_weight_cache_count++;
    return gpu_ptr;
}

static void weight_cache_clear(void) {
    for (int i = 0; i < g_weight_cache_count; i++) {
        cudaFree(g_weight_cache[i].gpu_ptr);
    }
    g_weight_cache_count = 0;
}

/* ========================================================================
 * CUDA Kernels (Device Functions)
 * ======================================================================== */

/* BF16 -> F32 conversion */
__global__ void bf16_to_f32_kernel(float *out, const uint16_t *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint16_t bits = in[idx];
    uint32_t sign = (uint32_t)(bits >> 15) << 31;
    uint32_t exp  = (uint32_t)((bits >> 7) & 0xFF);
    uint32_t mant = (uint32_t)(bits & 0x7F) << 16;
    if (exp == 0) exp = 0;
    else if (exp == 0xFF) exp = 0xFF;
    else exp += 112;
    union { float f; uint32_t i; } v;
    v.i = sign | (exp << 23) | mant;
    out[idx] = v.f;
}

/* AdaLN: out = (1 + scale) * rmsnorm(x) + shift */
__global__ void adaln_norm_kernel(float *out, const float *x,
                                  const float *shift, const float *scale,
                                  int seq, int hidden, float eps) {
    extern __shared__ float s_mem[];
    float *s_sq = s_mem;

    int h = threadIdx.x;
    int s = blockIdx.x;

    if (s >= seq || h >= hidden) return;

    int idx = s * hidden + h;

    /* Compute sum of squares for this sequence position */
    float sum_sq = 0.0f;
    for (int j = h; j < hidden; j += blockDim.x) {
        float v = x[s * hidden + j];
        sum_sq += v * v;
    }
    s_sq[h] = sum_sq;
    __syncthreads();

    if (h == 0) {
        float total = 0.0f;
        for (int j = 0; j < hidden; j++) total += s_sq[j];
        float rms = sqrtf(total / hidden + eps);
        s_sq[0] = 1.0f / rms;
    }
    __syncthreads();

    float rms_rcp = s_sq[0];
    float s_val = scale ? scale[h] : 1.0f;
    float sh_val = shift ? shift[h] : 0.0f;
    out[idx] = (1.0f + s_val) * (x[idx] * rms_rcp) + sh_val;
}

/* QK RMSNorm: separate norm on q and k with learned weights */
__global__ void qk_rms_norm_kernel(float *q, float *k,
                                   const float *wq, const float *wk,
                                   int seq, int heads, int head_dim, float eps) {
    int tid = threadIdx.x;
    int pos = blockIdx.x;     /* position index = seq * heads + head */
    int total = seq * heads;
    if (pos >= total) return;
    int dim = head_dim;

    extern __shared__ float s_mem[];
    float *s_sq = s_mem;

    float sum_sq_q = 0.0f, sum_sq_k = 0.0f;
    int base = pos * dim;

    for (int j = tid; j < dim; j += blockDim.x) {
        float vq = q[base + j];
        float vk = k[base + j];
        sum_sq_q += vq * vq;
        sum_sq_k += vk * vk;
    }
    s_sq[tid] = sum_sq_q;
    s_sq[tid + blockDim.x] = sum_sq_k;
    __syncthreads();

    if (tid == 0) {
        float total_q = 0.0f, total_k = 0.0f;
        for (int j = 0; j < blockDim.x; j++) {
            total_q += s_sq[j];
            total_k += s_sq[j + blockDim.x];
        }
        s_sq[0] = 1.0f / sqrtf(total_q / dim + eps);
        s_sq[1] = 1.0f / sqrtf(total_k / dim + eps);
    }
    __syncthreads();

    float rcp_q = s_sq[0];
    float rcp_k = s_sq[1];
    int h_idx = pos % heads;

    for (int j = tid; j < dim; j += blockDim.x) {
        int idx = base + j;
        float wq_val = wq ? wq[h_idx * dim + j] : 1.0f;
        float wk_val = wk ? wk[h_idx * dim + j] : 1.0f;
        q[idx] = q[idx] * rcp_q * wq_val;
        k[idx] = k[idx] * rcp_k * wk_val;
    }
}

/* Split fused QKV+MLP: [seq, h*3+mlp*2] -> q[seq,h], k[seq,h], v[seq,h], gate[seq,mlp], up[seq,mlp] */
__global__ void split_qkv_mlp_kernel(const float *fused,
                                     float *q, float *k, float *v,
                                     float *gate, float *up,
                                     int seq, int hidden, int mlp_hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * (hidden * 3 + mlp_hidden * 2);
    if (idx >= total) return;

    int pos = idx;
    int dim = hidden * 3 + mlp_hidden * 2;
    int s = pos / dim;
    int h = pos % dim;

    float val = fused[pos];

    if (h < hidden) {
        q[s * hidden + h] = val;
    } else if (h < 2 * hidden) {
        k[s * hidden + (h - hidden)] = val;
    } else if (h < 3 * hidden) {
        v[s * hidden + (h - 3 * hidden)] = val;
    } else if (h < 3 * hidden + mlp_hidden) {
        gate[s * mlp_hidden + (h - 3 * hidden)] = val;
    } else {
        up[s * mlp_hidden + (h - 3 * hidden - mlp_hidden)] = val;
    }
}

/* RoPE 2D (Flux style): out0 = cos*x0 - sin*x1, out1 = cos*x1 + sin*x0 */
__global__ void rope_2d_kernel(float *q, float *k,
                               const float *cos_freq, const float *sin_freq,
                               int seq, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim;
    if (idx >= total) return;

    int s = idx / (heads * head_dim);
    int h = (idx / head_dim) % heads;
    int d = idx % head_dim;

    /* RoPE pairs: (d, d+1) */
    if (d % 2 != 0) return;

    int freq_idx = s * (head_dim / 2) + d / 2;
    float cos_v = cos_freq[freq_idx];
    float sin_v = sin_freq[freq_idx];

    int base = s * heads * head_dim + h * head_dim;
    int i0 = base + d;
    int i1 = base + d + 1;

    float q0 = q[i0], q1 = q[i1];
    q[i0] = cos_v * q0 - sin_v * q1;
    q[i1] = cos_v * q1 + sin_v * q0;

    float k0 = k[i0], k1 = k[i1];
    k[i0] = cos_v * k0 - sin_v * k1;
    k[i1] = cos_v * k1 + sin_v * k0;
}

/* SiLU(gate) * up in-place on gate */
__global__ void silu_mul_kernel(float *gate, const float *up, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = gate[idx];
    float sig = 1.0f / (1.0f + expf(-x));
    gate[idx] = (x * sig) * up[idx];
}

/* Concat attn [seq,hidden] and mlp [seq,mlp_hidden] into out [seq,hidden+mlp_hidden] */
__global__ void concat_attn_mlp_kernel(const float *attn, const float *mlp,
                                       float *out, int seq, int hidden, int mlp_hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * (hidden + mlp_hidden);
    if (idx >= total) return;

    int s = idx / (hidden + mlp_hidden);
    int h = idx % (hidden + mlp_hidden);

    if (h < hidden) {
        out[idx] = attn[s * hidden + h];
    } else {
        out[idx] = mlp[s * mlp_hidden + (h - hidden)];
    }
}

/* Gated residual add: out += gate_val * proj_out */
__global__ void gated_add_kernel(float *out, const float *proj_out,
                                 const float *gate, int seq, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * hidden;
    if (idx >= total) return;

    int h = idx % hidden;
    out[idx] += gate[h] * proj_out[idx];
}

/* Softmax over last dimension (used by attention) */
__global__ void softmax_kernel(float *x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float *row_ptr = x + row * cols;

    float max_val = -1e20f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(row_ptr[i] - s_max);
    }

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_ptr[i] = expf(row_ptr[i] - s_max) / s_sum;
    }
}

/* ========================================================================
 * Public API Implementation
 * ======================================================================== */

int iris_cuda_init(void) {
    if (g_initialized) return 1;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return 0;

    cudaSetDevice(0);
    cudaStreamCreate(&g_stream);

    cublasStatus_t cs = cublasCreate(&g_cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
        return 0;
    }
    cublasSetStream(g_cublas, g_stream);

    /* Sanity check: run a tiny SGEMM to verify cuBLAS works */
    {
        float *d_a, *d_b, *d_c;
        float one = 1.0f, zero = 0.0f;
        if (cudaMalloc(&d_a, 16 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_b, 16 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_c, 4 * sizeof(float)) != cudaSuccess) {
            cublasDestroy(g_cublas);
            cudaStreamDestroy(g_stream);
            g_cublas = NULL; g_stream = NULL;
            return 0;
        }
        cublasStatus_t cs = cublasSgemm(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N, 4, 1, 4,
            &one, d_a, 4, d_b, 4, &zero, d_c, 4);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        if (cs != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS sanity check failed: %d\n", (int)cs);
            cublasDestroy(g_cublas);
            cudaStreamDestroy(g_stream);
            g_cublas = NULL; g_stream = NULL;
            return 0;
        }
    }

    g_initialized = 1;
    return 1;
}

int iris_cuda_available(void) {
    return g_initialized;
}

void iris_cuda_cleanup(void) {
    if (!g_initialized) return;

    weight_cache_clear();
    pool_destroy();

    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = NULL;
    }
    if (g_stream) {
        cudaStreamSynchronize(g_stream);
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
    }
    g_initialized = 0;
    g_batch_mode = 0;
}

void iris_cuda_reset_transient(void) {
    pool_reset_transient();
    g_batch_mode = 0;
}

void iris_cuda_sync(void) {
    if (g_stream) cudaStreamSynchronize(g_stream);
}

void iris_cuda_begin_batch(void) {
    g_batch_mode = 1;
}

void iris_cuda_end_batch(void) {
    if (g_batch_mode) {
        cudaStreamSynchronize(g_stream);
        g_batch_mode = 0;
    }
}

int iris_cuda_in_batch(void) {
    return g_batch_mode;
}

/* ========================================================================
 * Tensor API
 * ======================================================================== */

iris_cuda_tensor_t iris_cuda_tensor_create(const float *data, size_t num_elements) {
    if (!g_initialized || !data || num_elements == 0) return NULL;

    float *dptr = NULL;
    cudaMalloc(&dptr, num_elements * sizeof(float));
    if (!dptr) return NULL;

    cudaMemcpy(dptr, data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    iris_cuda_tensor_t t = (iris_cuda_tensor_t)malloc(sizeof(struct iris_cuda_tensor));
    if (!t) { cudaFree(dptr); return NULL; }
    t->device_ptr = dptr;
    t->num_elements = num_elements;
    t->has_pending_work = 0;
    t->persistent = 0;
    return t;
}

iris_cuda_tensor_t iris_cuda_tensor_alloc(size_t num_elements) {
    if (!g_initialized || num_elements == 0) return NULL;

    float *dptr = pool_alloc(num_elements);
    if (!dptr) {
        cudaMalloc(&dptr, num_elements * sizeof(float));
        if (!dptr) return NULL;
    }

    iris_cuda_tensor_t t = (iris_cuda_tensor_t)malloc(sizeof(struct iris_cuda_tensor));
    if (!t) { pool_release(dptr); return NULL; }
    t->device_ptr = dptr;
    t->num_elements = num_elements;
    t->has_pending_work = 0;
    t->persistent = 0;
    return t;
}

iris_cuda_tensor_t iris_cuda_tensor_alloc_persistent(size_t num_elements) {
    iris_cuda_tensor_t t = iris_cuda_tensor_alloc(num_elements);
    if (t) t->persistent = 1;
    return t;
}

void iris_cuda_tensor_read(iris_cuda_tensor_t tensor, float *out) {
    if (!tensor || !out) return;
    if (tensor->has_pending_work) {
        iris_cuda_sync();
        tensor->has_pending_work = 0;
    }
    cudaMemcpy(out, tensor->device_ptr, tensor->num_elements * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void iris_cuda_tensor_write(iris_cuda_tensor_t tensor, const float *data) {
    if (!tensor || !data) return;
    if (tensor->has_pending_work) {
        iris_cuda_sync();
        tensor->has_pending_work = 0;
    }
    cudaMemcpy(tensor->device_ptr, data, tensor->num_elements * sizeof(float),
               cudaMemcpyHostToDevice);
}

void iris_cuda_tensor_free(iris_cuda_tensor_t tensor) {
    if (!tensor) return;
    if (tensor->persistent) {
        cudaFree(tensor->device_ptr);
    } else {
        pool_release(tensor->device_ptr);
    }
    free(tensor);
}

float *iris_cuda_tensor_data(iris_cuda_tensor_t tensor) {
    if (!tensor) return NULL;
    return tensor->device_ptr;
}

size_t iris_cuda_tensor_size(iris_cuda_tensor_t tensor) {
    if (!tensor) return 0;
    return tensor->num_elements;
}

/* ========================================================================
 * Tensor Operations
 * ======================================================================== */

int iris_cuda_linear_into(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                          const float *W, const float *b,
                          int seq_len, int in_dim, int out_dim) {
    if (!g_initialized || !out || !x || !W) return 0;
    if ((size_t)seq_len * out_dim > out->num_elements) return 0;
    if ((size_t)seq_len * in_dim > x->num_elements) return 0;

    float *W_gpu = weight_cache_get(W, (size_t)in_dim * out_dim, 0);
    if (!W_gpu) return 0;

    float *d_out = out->device_ptr;
    float *d_x = x->device_ptr;

    /* out[seq,out_dim] = x[seq,in_dim] @ W[out_dim,in_dim]^T (row-major)
     * cuBLAS column-major: y^T[out_dim,seq] = W^T_col[in_dim,out_dim]^T × x^T[in_dim,seq]
     *                    = W[out_dim,in_dim] × x^T[in_dim,seq]
     * Using CUBLAS_OP_T on W: W stored as [in_dim×out_dim] col-major, ld=in_dim
     * Using CUBLAS_OP_N on x: x stored as [in_dim×seq] col-major, ld=in_dim */
    float alpha = 1.0f, beta = 0.0f;
    if (!cublas_check(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  out_dim, seq_len, in_dim,
                                  &alpha,
                                  W_gpu, in_dim,
                                  d_x, in_dim,
                                  &beta,
                                  d_out, out_dim),
                      "linear sgemm")) return 0;

    if (b) {
        float *b_gpu = weight_cache_get(b, out_dim, 0);
        if (b_gpu) {
            for (int s = 0; s < seq_len; s++) {
                if (!cublas_check(cublasSaxpy(g_cublas, out_dim, &alpha,
                                              b_gpu, 1,
                                              d_out + s * out_dim, 1),
                                  "linear bias add")) return 0;
            }
        }
    }

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_linear_bf16_into(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                               const uint16_t *W_bf16,
                               int seq_len, int in_dim, int out_dim) {
    if (!g_initialized || !out || !x || !W_bf16) return 0;
    if ((size_t)seq_len * out_dim > out->num_elements) return 0;
    if ((size_t)seq_len * in_dim > x->num_elements) return 0;

    float *W_gpu = weight_cache_get(W_bf16, (size_t)in_dim * out_dim, 1);
    if (!W_gpu) return 0;

    float *d_out = out->device_ptr;
    float *d_x = x->device_ptr;

    float alpha = 1.0f, beta = 0.0f;
    if (!cublas_check(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  out_dim, seq_len, in_dim,
                                  &alpha,
                                  W_gpu, in_dim,
                                  d_x, in_dim,
                                  &beta,
                                  d_out, out_dim),
                      "linear bf16 sgemm")) return 0;

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_adaln_norm(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                         const float *shift, const float *scale,
                         int seq, int hidden, float eps) {
    if (!g_initialized || !out || !x) return 0;
    if (out->num_elements < (size_t)seq * hidden) return 0;
    if (x->num_elements < (size_t)seq * hidden) return 0;

    int shared_mem = hidden * sizeof(float);
    int max_shared = 49152;  /* 48 KB typical limit */
    int block_size = hidden < 1024 ? hidden : (hidden < 2048 ? 512 : 256);

    if (shared_mem > max_shared) {
        /* Fallback: sequential reduction */
        dim3 block(256);
        dim3 grid(seq);
        adaln_norm_kernel<<<grid, block, max_shared, g_stream>>>(
            out->device_ptr, x->device_ptr, shift, scale, seq, hidden, eps);
    } else {
        dim3 block(block_size);
        dim3 grid(seq);
        adaln_norm_kernel<<<grid, block, shared_mem, g_stream>>>(
            out->device_ptr, x->device_ptr, shift, scale, seq, hidden, eps);
    }

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_qk_rms_norm(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                          const float *weight_q, const float *weight_k,
                          int seq, int heads, int head_dim, float eps) {
    if (!g_initialized || !q || !k) return 0;
    int total_pos = seq * heads;
    int dim = head_dim;
    int block_size = dim < 256 ? dim : 256;
    int shared = (block_size * 2) * sizeof(float);

    dim3 block(block_size);
    dim3 grid(total_pos);
    qk_rms_norm_kernel<<<grid, block, shared, g_stream>>>(
        q->device_ptr, k->device_ptr, weight_q, weight_k,
        seq, heads, head_dim, eps);

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_split_qkv_mlp(iris_cuda_tensor_t fused,
                            iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                            iris_cuda_tensor_t v,
                            iris_cuda_tensor_t gate, iris_cuda_tensor_t up,
                            int seq, int hidden, int mlp_hidden) {
    if (!g_initialized || !fused || !q || !k || !v || !gate || !up) return 0;

    int total = seq * (hidden * 3 + mlp_hidden * 2);
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    split_qkv_mlp_kernel<<<grid, block, 0, g_stream>>>(
        fused->device_ptr, q->device_ptr, k->device_ptr, v->device_ptr,
        gate->device_ptr, up->device_ptr,
        seq, hidden, mlp_hidden);

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_rope_2d(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                      const float *cos_freq, const float *sin_freq,
                      int seq, int heads, int head_dim) {
    if (!g_initialized || !q || !k) return 0;

    int total = seq * heads * head_dim;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    /* Upload cos/sin freq to GPU if not already cached */
    size_t freq_elems = (size_t)seq * (head_dim / 2);
    float *cos_gpu = weight_cache_get(cos_freq, freq_elems, 0);
    float *sin_gpu = weight_cache_get(sin_freq, freq_elems, 0);

    rope_2d_kernel<<<grid, block, 0, g_stream>>>(
        q->device_ptr, k->device_ptr,
        cos_gpu ? cos_gpu : cos_freq,
        sin_gpu ? sin_gpu : sin_freq,
        seq, heads, head_dim);

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_attention(iris_cuda_tensor_t out,
                        iris_cuda_tensor_t Q, iris_cuda_tensor_t K,
                        iris_cuda_tensor_t V,
                        int seq, int heads, int head_dim, float scale) {
    if (!g_initialized || !out || !Q || !K || !V) return 0;

    float *d_Q = Q->device_ptr;
    float *d_K = K->device_ptr;
    float *d_V = V->device_ptr;
    float *d_out = out->device_ptr;

    /* Allocate temp for ONE head's attention scores [seq, seq].
     * Process heads sequentially to reduce peak VRAM from
     * heads * seq * seq to seq * seq (e.g. 1.1 GB vs 26.5 GB at 1024x1024). */
    float *d_scores = NULL;
    {
        cudaError_t ce = cudaMalloc(&d_scores, (size_t)seq * seq * sizeof(float));
        if (ce != cudaSuccess) return 0;
    }

    float alpha_qk = scale;
    float alpha_v = 1.0f;
    float beta = 0.0f;
    int ok = 1;

    for (int h = 0; h < heads && ok; h++) {
        float *q_head = d_Q + h * head_dim;
        float *k_head = d_K + h * head_dim;
        float *v_head = d_V + h * head_dim;
        float *o_head = d_out + h * head_dim;

        if (!cublas_check(cublasSgemm(g_cublas,
                                      CUBLAS_OP_N, CUBLAS_OP_T,
                                      seq, seq, head_dim,
                                      &alpha_qk,
                                      k_head, seq,
                                      q_head, seq,
                                      &beta,
                                      d_scores, seq),
                          "attn QK")) { ok = 0; break; }

        dim3 sm_grid(seq);
        dim3 sm_block(256);
        softmax_kernel<<<sm_grid, sm_block, 0, g_stream>>>(d_scores, seq, seq);

        if (!cublas_check(cublasSgemm(g_cublas,
                                      CUBLAS_OP_T, CUBLAS_OP_T,
                                      head_dim, seq, seq,
                                      &alpha_v,
                                      v_head, head_dim,
                                      d_scores, seq,
                                      &beta,
                                      o_head, head_dim),
                          "attn V")) { ok = 0; break; }
    }

    cudaFree(d_scores);
    if (!ok) return 0;

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_silu_mul(iris_cuda_tensor_t gate, iris_cuda_tensor_t up, int n) {
    if (!g_initialized || !gate || !up) return 0;

    dim3 block(256);
    dim3 grid((n + 255) / 256);
    silu_mul_kernel<<<grid, block, 0, g_stream>>>(gate->device_ptr, up->device_ptr, n);

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_concat_attn_mlp(iris_cuda_tensor_t attn, iris_cuda_tensor_t mlp,
                              iris_cuda_tensor_t out,
                              int seq, int hidden, int mlp_hidden) {
    if (!g_initialized || !attn || !mlp || !out) return 0;

    int total = seq * (hidden + mlp_hidden);
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    concat_attn_mlp_kernel<<<grid, block, 0, g_stream>>>(
        attn->device_ptr, mlp->device_ptr, out->device_ptr,
        seq, hidden, mlp_hidden);

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}

int iris_cuda_gated_add(iris_cuda_tensor_t out, iris_cuda_tensor_t proj_out,
                        const float *gate, int seq, int hidden) {
    if (!g_initialized || !out || !proj_out || !gate) return 0;

    float *gate_gpu = weight_cache_get(gate, hidden, 0);

    dim3 block(256);
    dim3 grid((seq * hidden + 255) / 256);
    gated_add_kernel<<<grid, block, 0, g_stream>>>(
        out->device_ptr, proj_out->device_ptr,
        gate_gpu ? gate_gpu : gate,
        seq, hidden);

    if (!g_batch_mode) iris_cuda_sync();
    return 1;
}
