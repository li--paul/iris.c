#include "iris_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static cublasHandle_t cuda_handle = NULL;
static int cuda_initialized = 0;
static int cuda_in_batch = 0;

/* Minimum matrix size to use GPU (smaller is faster on CPU) */
#define MIN_CUDA_ELEMENTS (256 * 256)

/* GPU memory pointers for cached weights */
#define MAX_CACHED_WEIGHTS 256
static struct {
    const float *cpu_ptr;
    float *gpu_ptr;
    size_t size;
} weight_cache[MAX_CACHED_WEIGHTS];
static int num_cached = 0;

static int cuda_check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR [%s]: %s\n", msg, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

static int cublas_check(cublasStatus_t err, const char *msg) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS ERROR [%s]: %d\n", msg, (int)err);
        return 0;
    }
    return 1;
}

static float *cuda_upload(const float *src, size_t n) {
    float *dst = NULL;
    if (!cuda_check(cudaMalloc((void**)&dst, n * sizeof(float)), "upload malloc")) return NULL;
    if (src && !cuda_check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice), "upload memcpy")) {
        cudaFree(dst); return NULL;
    }
    return dst;
}

static void cuda_download(float *dst, const float *src, size_t n) {
    cuda_check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost), "download memcpy");
}

/* Row-major → cuBLAS column-major adapter.
   cublasSgemm(handle, transb, transa, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc)
   computes: C[N,M] = op(B)[N,K] * op(A)[K,M]
   which equals (op(A) * op(B))^T, matching row-major output. */
static void cuda_gemm(int transa, int transb,
                      int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc) {
    if (!cuda_handle) return;
    cublasOperation_t cu_transb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cu_transa = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(cuda_handle, cu_transb, cu_transa,
                N, M, K,
                &alpha,
                B, ldb,
                A, lda,
                &beta,
                C, ldc);
}

int iris_cuda_init(void) {
    if (cuda_initialized) return 1;
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) return 0;
    cublasStatus_t cs = cublasCreate(&cuda_handle);
    if (cs != CUBLAS_STATUS_SUCCESS) { cuda_handle = NULL; return 0; }
    cuda_initialized = 1;
    return 1;
}

int iris_cuda_available(void) {
    return cuda_initialized && cuda_handle != NULL;
}

void iris_cuda_cleanup(void) {
    if (cuda_handle) { cublasDestroy(cuda_handle); cuda_handle = NULL; }
    for (int i = 0; i < num_cached; i++) {
        if (weight_cache[i].gpu_ptr) cudaFree(weight_cache[i].gpu_ptr);
    }
    num_cached = 0;
    cuda_initialized = 0;
}

void iris_cuda_sync(void) {
    if (cuda_handle) cudaDeviceSynchronize();
}

/* ------------------------------------------------------------------ */
/*  SGEMM wrapper                                                      */
/* ------------------------------------------------------------------ */

void iris_cuda_sgemm(int transpose_a, int transpose_b,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc) {
    float *dA = cuda_upload(A, (size_t)M * K);
    if (!dA) return;

    float *dB = cuda_upload(B, (size_t)K * N);
    if (!dB) { cudaFree(dA); return; }

    float *dC = NULL;
    if (beta == 0.0f) {
        cudaMalloc((void**)&dC, (size_t)M * N * sizeof(float));
    } else {
        dC = cuda_upload(C, (size_t)M * N);
    }
    if (!dC) { cudaFree(dA); cudaFree(dB); return; }

    cuda_gemm(transpose_a, transpose_b, M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc);

    cuda_download(C, dC, (size_t)M * N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

/* Cached variant: keeps B on GPU for reuse across calls */
void iris_cuda_sgemm_cached(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            float beta,
                            float *C, int ldc) {
    /* Look up or upload B */
    float *dB = NULL;
    int found = 0;
    for (int i = 0; i < num_cached; i++) {
        if (weight_cache[i].cpu_ptr == B) {
            dB = weight_cache[i].gpu_ptr;
            found = 1;
            break;
        }
    }
    if (!found) {
        dB = cuda_upload(B, (size_t)K * N);
        if (dB && num_cached < MAX_CACHED_WEIGHTS) {
            weight_cache[num_cached].cpu_ptr = B;
            weight_cache[num_cached].gpu_ptr = dB;
            weight_cache[num_cached].size = (size_t)K * N;
            num_cached++;
        }
    }

    if (!dB) return; /* upload failed */

    float *dA = cuda_upload(A, (size_t)M * K);
    if (!dA) return;

    float *dC = NULL;
    if (beta == 0.0f) {
        cudaMalloc((void**)&dC, (size_t)M * N * sizeof(float));
    } else {
        dC = cuda_upload(C, (size_t)M * N);
    }
    if (!dC) { cudaFree(dA); return; }

    cuda_gemm(transpose_a, transpose_b, M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc);

    cuda_download(C, dC, (size_t)M * N);
    cudaFree(dA); cudaFree(dC);
}

/* ------------------------------------------------------------------ */
/*  Linear layer: y = x @ W^T + b                                      */
/* ------------------------------------------------------------------ */

void iris_cuda_linear(float *y, const float *x, const float *W, const float *b,
                      int seq_len, int in_dim, int out_dim) {
    size_t elements = (size_t)seq_len * out_dim;
    if (elements < MIN_CUDA_ELEMENTS) return;

    float *dx = cuda_upload(x, (size_t)seq_len * in_dim);
    if (!dx) return;

    /* W is [out_dim, in_dim], we need W^T for the matmul.
       cuda_gemm with transpose_b=1 handles this. */
    float *dW = cuda_upload(W, (size_t)out_dim * in_dim);
    if (!dW) { cudaFree(dx); return; }

    float *dy = NULL;
    cudaMalloc((void**)&dy, (size_t)seq_len * out_dim * sizeof(float));
    if (!dy) { cudaFree(dx); cudaFree(dW); return; }

    cuda_gemm(0, 1, seq_len, out_dim, in_dim, 1.0f, dx, in_dim, dW, in_dim, 0.0f, dy, out_dim);

    if (b) {
        /* Add bias: for each seq, add b[0..out_dim) */
        float *db = cuda_upload(b, out_dim);
        if (db) {
            for (int s = 0; s < seq_len; s++) {
                cublasSaxpy(cuda_handle, out_dim, &(float){1.0f}, db, 1, dy + s * out_dim, 1);
            }
            cudaFree(db);
        }
    }

    cuda_download(y, dy, (size_t)seq_len * out_dim);
    cudaFree(dx); cudaFree(dW); cudaFree(dy);
}

/* ------------------------------------------------------------------ */
/*  Convolution via im2col + GEMM                                      */
/* ------------------------------------------------------------------ */

void iris_cuda_conv2d(float *out, const float *in, const float *weight, const float *bias,
                      int batch, int in_ch, int out_ch, int H, int W,
                      int kH, int kW, int stride, int padding) {
    int outH = (H + 2 * padding - kH) / stride + 1;
    int outW = (W + 2 * padding - kW) / stride + 1;
    int K = in_ch * kH * kW;
    int tile_pixels = outH * outW;

    size_t col_n = (size_t)K * tile_pixels;
    if (col_n < MIN_CUDA_ELEMENTS) return; /* fall back to CPU */

    float *col = (float *)malloc(col_n * sizeof(float));
    if (!col) return;

    /* im2col on CPU, GEMM on GPU */
    for (int ic = 0; ic < in_ch; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int col_row = ic * kH * kW + kh * kW + kw;
                for (int oh = 0; oh < outH; oh++) {
                    int ih = oh * stride - padding + kh;
                    for (int ow = 0; ow < outW; ow++) {
                        int iw = ow * stride - padding + kw;
                        int col_idx = col_row * tile_pixels + oh * outW + ow;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            col[col_idx] = in[ic * H * W + ih * W + iw];
                        else
                            col[col_idx] = 0.0f;
                    }
                }
            }
        }
    }

    /* Upload col and weight to GPU */
    float *dcol = cuda_upload(col, col_n);
    free(col);
    if (!dcol) return;

    float *dw = cuda_upload(weight, (size_t)out_ch * K);
    if (!dw) { cudaFree(dcol); return; }

    float *dout = NULL;
    cudaMalloc((void**)&dout, (size_t)batch * out_ch * tile_pixels * sizeof(float));
    if (!dout) { cudaFree(dcol); cudaFree(dw); return; }

    for (int b = 0; b < batch; b++) {
        float *dout_b = dout + b * out_ch * tile_pixels;
        cuda_gemm(0, 0, out_ch, tile_pixels, K,
                  1.0f, dw, K, dcol, tile_pixels, 0.0f, dout_b, tile_pixels);
    }

    cuda_download(out, dout, (size_t)batch * out_ch * tile_pixels);

    if (bias) {
        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float b_val = bias[oc];
                float *out_boc = out + (b * out_ch + oc) * tile_pixels;
                for (int i = 0; i < tile_pixels; i++) out_boc[i] += b_val;
            }
        }
    }

    cudaFree(dcol); cudaFree(dw); cudaFree(dout);
}

/* --------------------------------------------------------------- */
/*  Multi-head attention: handles the [seq, heads*head_dim] layout  */
/* --------------------------------------------------------------- */

int iris_cuda_attention(float *out,
                        const float *Q, const float *K, const float *V,
                        int seq_q, int seq_k, int heads, int head_dim,
                        float scale) {
    int hidden = heads * head_dim;

    size_t head_size = (size_t)seq_q * head_dim;
    size_t scores_size = (size_t)seq_q * seq_k;
    float *q_t = (float *)malloc(heads * head_size * sizeof(float));
    float *k_t = (float *)malloc(heads * head_size * sizeof(float));
    float *v_t = (float *)malloc(heads * (size_t)seq_k * head_dim * sizeof(float));
    float *scores = (float *)malloc(heads * scores_size * sizeof(float));
    float *out_t = (float *)malloc(heads * head_size * sizeof(float));

    if (!q_t || !k_t || !v_t || !scores || !out_t) {
        free(q_t); free(k_t); free(v_t); free(scores); free(out_t);
        return 0;
    }

    /* Q: [seq_q, heads*head_dim] → [heads, seq_q, head_dim] */
    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seq_q; s++) {
            for (int d = 0; d < head_dim; d++) {
                q_t[h * head_size + s * head_dim + d] = Q[s * hidden + h * head_dim + d];
            }
        }
    }
    /* K, V: [seq_k, heads*head_dim] → [heads, seq_k, head_dim] */
    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seq_k; s++) {
            for (int d = 0; d < head_dim; d++) {
                k_t[h * (size_t)seq_k * head_dim + s * head_dim + d] = K[s * hidden + h * head_dim + d];
                v_t[h * (size_t)seq_k * head_dim + s * head_dim + d] = V[s * hidden + h * head_dim + d];
            }
        }
    }

    /* Allocate all GPU buffers in one chunk to avoid WSL driver heap issue */
    size_t k_size = heads * (size_t)seq_k * head_dim;
    size_t total_gpu = (heads * head_size * 3) + (heads * scores_size) + k_size;
    float *gpu_buf = NULL;
    cudaError_t ce = cudaMalloc((void**)&gpu_buf, total_gpu * sizeof(float));
    if (ce != cudaSuccess || !gpu_buf) {
        free(q_t); free(k_t); free(v_t); free(scores); free(out_t);
        return 0;
    }
    float *dq = gpu_buf;
    float *do_t = dq + heads * head_size;
    float *ds = do_t + heads * head_size;
    float *dv = ds + heads * scores_size;
    float *dk = dv + k_size;

    cudaMemcpy(dq, q_t, heads * head_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k_t, k_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_t, k_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scores, heads * scores_size * sizeof(float), cudaMemcpyHostToDevice);

    /* Per-head cuBLAS calls (avoids cublasSgemmStridedBatched bug in WSL driver) */
    for (int h = 0; h < heads; h++) {
        float *qh = dq + h * head_size;
        float *kh = dk + h * (size_t)seq_k * head_dim;
        float *vh = dv + h * (size_t)seq_k * head_dim;
        float *sh = ds + h * scores_size;
        float *oh = do_t + h * head_size;

        /* scores_h[seq_q, seq_k] = Q_h[seq_q, head_dim] @ K_h^T[head_dim, seq_k] */
        cublasSgemm(cuda_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_q, seq_k, head_dim,
                    &scale,
                    qh, head_dim,
                    kh, head_dim,
                    &(float){0.0f},
                    sh, seq_k);
    }

    iris_cuda_sync();

    /* Download scores, softmax on CPU, upload back */
    cuda_download(scores, ds, heads * scores_size);

    for (size_t h = 0; h < (size_t)heads; h++) {
        float *sh = scores + h * scores_size;
        for (int i = 0; i < seq_q; i++) {
            float *row = sh + i * seq_k;
            float max_val = row[0];
            for (int j = 1; j < seq_k; j++) if (row[j] > max_val) max_val = row[j];

            float sum = 0.0f;
            for (int j = 0; j < seq_k; j++) { row[j] = expf(row[j] - max_val); sum += row[j]; }
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < seq_k; j++) row[j] *= inv_sum;
        }
    }

    cudaMemcpy(ds, scores, heads * scores_size * sizeof(float), cudaMemcpyHostToDevice);

    /* out_h[seq_q, head_dim] = scores_h[seq_q, seq_k] @ V_h[seq_k, head_dim] */
    for (int h = 0; h < heads; h++) {
        float *sh = ds + h * scores_size;
        float *vh = dv + h * (size_t)seq_k * head_dim;
        float *oh = do_t + h * head_size;

        cublasSgemm(cuda_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    seq_q, head_dim, seq_k,
                    &(float){1.0f},
                    sh, seq_k,
                    vh, head_dim,
                    &(float){0.0f},
                    oh, head_dim);
    }

    iris_cuda_sync();

    /* Download and transpose back HSD → SHD */
    cuda_download(out_t, do_t, heads * head_size);

    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seq_q; s++) {
            for (int d = 0; d < head_dim; d++) {
                out[s * hidden + h * head_dim + d] = out_t[h * head_size + s * head_dim + d];
            }
        }
    }

    cudaFree(gpu_buf);
    free(q_t); free(k_t); free(v_t); free(scores); free(out_t);
    return 1;
}

const char *iris_cuda_device_name(void) {
    if (!cuda_initialized) return NULL;
    static char name[256];
    struct cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        snprintf(name, sizeof(name), "%s | %.0f MB VRAM | SM%d%d",
                 props.name, props.totalGlobalMem / 1048576.0f,
                 props.major, props.minor);
        return name;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Batch operations (no-ops for v1 — individual calls sync each op)   */
/* ------------------------------------------------------------------ */

void iris_cuda_begin_batch(void) { cuda_in_batch = 1; }
void iris_cuda_end_batch(void) { cuda_in_batch = 0; iris_cuda_sync(); }
