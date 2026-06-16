#include "iris_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int iris_cuda_silu_mul_device(float *d_gate, const float *d_up, int n);
extern int iris_cuda_rms_norm_device(float *d_out, const float *d_x, const float *d_weight,
                                     int rows, int hidden, float eps);
extern int iris_cuda_adaln_norm_device(float *d_out, const float *d_x,
                                       const float *d_shift, const float *d_scale,
                                       int rows, int hidden, float eps);
extern int iris_cuda_apply_rope_device(float *d_x, const float *d_freqs,
                                       int batch, int seq, int heads, int head_dim);

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

struct iris_cuda_tensor {
    float *ptr;
    size_t size;
};

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

static float *cuda_get_cached_weight(const float *src, size_t n) {
    if (!src || n == 0) return NULL;
    for (int i = 0; i < num_cached; i++) {
        if (weight_cache[i].cpu_ptr == src && weight_cache[i].size == n) {
            return weight_cache[i].gpu_ptr;
        }
    }

    float *dst = cuda_upload(src, n);
    if (!dst) return NULL;
    if (num_cached < MAX_CACHED_WEIGHTS) {
        weight_cache[num_cached].cpu_ptr = src;
        weight_cache[num_cached].gpu_ptr = dst;
        weight_cache[num_cached].size = n;
        num_cached++;
    }
    return dst;
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

iris_cuda_tensor_t iris_cuda_tensor_alloc(size_t num_elements) {
    if (num_elements == 0) return NULL;
    iris_cuda_tensor_t t = (iris_cuda_tensor_t)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_check(cudaMalloc((void**)&t->ptr, num_elements * sizeof(float)), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->size = num_elements;
    return t;
}

iris_cuda_tensor_t iris_cuda_tensor_create(const float *data, size_t num_elements) {
    iris_cuda_tensor_t t = iris_cuda_tensor_alloc(num_elements);
    if (!t) return NULL;
    if (data && !cuda_check(cudaMemcpy(t->ptr, data, num_elements * sizeof(float),
                                       cudaMemcpyHostToDevice), "tensor create memcpy")) {
        iris_cuda_tensor_free(t);
        return NULL;
    }
    return t;
}

void iris_cuda_tensor_free(iris_cuda_tensor_t tensor) {
    if (!tensor) return;
    if (tensor->ptr) cudaFree(tensor->ptr);
    free(tensor);
}

int iris_cuda_tensor_read(iris_cuda_tensor_t tensor, float *out) {
    if (!tensor || !tensor->ptr || !out) return 0;
    return cuda_check(cudaMemcpy(out, tensor->ptr, tensor->size * sizeof(float),
                                 cudaMemcpyDeviceToHost), "tensor read");
}

int iris_cuda_tensor_write(iris_cuda_tensor_t tensor, const float *data) {
    if (!tensor || !tensor->ptr || !data) return 0;
    return cuda_check(cudaMemcpy(tensor->ptr, data, tensor->size * sizeof(float),
                                 cudaMemcpyHostToDevice), "tensor write");
}

float *iris_cuda_tensor_data(iris_cuda_tensor_t tensor) {
    return tensor ? tensor->ptr : NULL;
}

size_t iris_cuda_tensor_size(iris_cuda_tensor_t tensor) {
    return tensor ? tensor->size : 0;
}

int iris_cuda_linear_tensor(iris_cuda_tensor_t out, iris_cuda_tensor_t x,
                            const float *W, int seq_len, int in_dim, int out_dim,
                            int cache_weight) {
    if (!out || !x || !out->ptr || !x->ptr || !W) return 0;
    if (x->size < (size_t)seq_len * in_dim || out->size < (size_t)seq_len * out_dim) return 0;

    size_t w_size = (size_t)out_dim * in_dim;
    float *dW = cache_weight ? cuda_get_cached_weight(W, w_size) : cuda_upload(W, w_size);
    if (!dW) return 0;

    cuda_gemm(0, 1, seq_len, out_dim, in_dim, 1.0f,
              x->ptr, in_dim, dW, in_dim, 0.0f, out->ptr, out_dim);
    if (!cache_weight) cudaFree(dW);
    return 1;
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
        float *db = cuda_upload(b, out_dim);
        if (db) {
            iris_cuda_add_bias_rows(dy, db, seq_len, out_dim);
            cudaFree(db);
        }
    }

    cuda_download(y, dy, (size_t)seq_len * out_dim);
    cudaFree(dx); cudaFree(dW); cudaFree(dy);
}

int iris_cuda_rms_norm(float *out, const float *x, const float *weight,
                       int seq_len, int hidden, float eps) {
    size_t elements = (size_t)seq_len * hidden;
    float *dx = cuda_upload(x, elements);
    if (!dx) return 0;
    float *dw = cuda_upload(weight, hidden);
    if (!dw) { cudaFree(dx); return 0; }
    float *dy = NULL;
    if (!cuda_check(cudaMalloc((void**)&dy, elements * sizeof(float)), "rms norm malloc")) {
        cudaFree(dx); cudaFree(dw);
        return 0;
    }

    int ok = iris_cuda_rms_norm_device(dy, dx, dw, seq_len, hidden, eps);
    if (ok) cuda_download(out, dy, elements);

    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    return ok;
}

int iris_cuda_adaln_norm(float *out, const float *x,
                         const float *shift, const float *scale,
                         int seq_len, int hidden, float eps) {
    size_t elements = (size_t)seq_len * hidden;
    float *dx = cuda_upload(x, elements);
    if (!dx) return 0;
    float *dshift = cuda_upload(shift, hidden);
    if (!dshift) { cudaFree(dx); return 0; }
    float *dscale = cuda_upload(scale, hidden);
    if (!dscale) { cudaFree(dx); cudaFree(dshift); return 0; }
    float *dy = NULL;
    if (!cuda_check(cudaMalloc((void**)&dy, elements * sizeof(float)), "adaln malloc")) {
        cudaFree(dx); cudaFree(dshift); cudaFree(dscale);
        return 0;
    }

    int ok = iris_cuda_adaln_norm_device(dy, dx, dshift, dscale, seq_len, hidden, eps);
    if (ok) cuda_download(out, dy, elements);

    cudaFree(dx); cudaFree(dshift); cudaFree(dscale); cudaFree(dy);
    return ok;
}

int iris_cuda_silu_mul(float *gate, const float *up, int n) {
    float *dg = cuda_upload(gate, n);
    if (!dg) return 0;
    float *du = cuda_upload(up, n);
    if (!du) { cudaFree(dg); return 0; }

    int ok = iris_cuda_silu_mul_device(dg, du, n);
    if (ok) cuda_download(gate, dg, n);

    cudaFree(dg); cudaFree(du);
    return ok;
}

int iris_cuda_apply_rope(float *x, const float *freqs,
                         int batch, int seq, int heads, int head_dim) {
    size_t elements = (size_t)batch * seq * heads * head_dim;
    size_t freq_elements = (size_t)seq * (head_dim / 2) * 2;
    float *dx = cuda_upload(x, elements);
    if (!dx) return 0;
    float *df = cuda_upload(freqs, freq_elements);
    if (!df) { cudaFree(dx); return 0; }

    int ok = iris_cuda_apply_rope_device(dx, df, batch, seq, heads, head_dim);
    if (ok) cuda_download(x, dx, elements);

    cudaFree(dx); cudaFree(df);
    return ok;
}

int iris_cuda_split_qkv_mlp(iris_cuda_tensor_t fused,
                            iris_cuda_tensor_t q, iris_cuda_tensor_t k, iris_cuda_tensor_t v,
                            iris_cuda_tensor_t gate, iris_cuda_tensor_t up,
                            int seq, int hidden, int mlp_hidden) {
    if (!fused || !q || !k || !v || !gate || !up) return 0;
    return iris_cuda_split_qkv_mlp_device(fused->ptr, q->ptr, k->ptr, v->ptr,
                                          gate->ptr, up->ptr, seq, hidden, mlp_hidden);
}

int iris_cuda_qk_rms_norm(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps) {
    if (!q || !k || !q_weight || !k_weight) return 0;
    float *dq_weight = cuda_upload(q_weight, head_dim);
    if (!dq_weight) return 0;
    float *dk_weight = cuda_upload(k_weight, head_dim);
    if (!dk_weight) { cudaFree(dq_weight); return 0; }
    int ok = iris_cuda_qk_rms_norm_device(q->ptr, k->ptr, dq_weight, dk_weight,
                                          seq, heads, head_dim, eps);
    cudaFree(dq_weight);
    cudaFree(dk_weight);
    return ok;
}

int iris_cuda_rope_unified(iris_cuda_tensor_t q, iris_cuda_tensor_t k,
                           const float *txt_cos, const float *txt_sin,
                           const float *img_cos, const float *img_sin,
                           int seq, int img_offset, int heads, int head_dim) {
    if (!q || !k || !txt_cos || !txt_sin || !img_cos || !img_sin) return 0;
    int img_seq = seq - img_offset;
    if (img_seq <= 0 || img_offset <= 0) return 0;
    float *dtxt_cos = cuda_upload(txt_cos, (size_t)img_offset * head_dim);
    if (!dtxt_cos) return 0;
    float *dtxt_sin = cuda_upload(txt_sin, (size_t)img_offset * head_dim);
    if (!dtxt_sin) { cudaFree(dtxt_cos); return 0; }
    float *dimg_cos = cuda_upload(img_cos, (size_t)img_seq * head_dim);
    if (!dimg_cos) { cudaFree(dtxt_cos); cudaFree(dtxt_sin); return 0; }
    float *dimg_sin = cuda_upload(img_sin, (size_t)img_seq * head_dim);
    if (!dimg_sin) { cudaFree(dtxt_cos); cudaFree(dtxt_sin); cudaFree(dimg_cos); return 0; }

    int ok = iris_cuda_rope_unified_device(q->ptr, k->ptr, dtxt_cos, dtxt_sin,
                                           dimg_cos, dimg_sin,
                                           seq, img_offset, heads, head_dim);
    cudaFree(dtxt_cos);
    cudaFree(dtxt_sin);
    cudaFree(dimg_cos);
    cudaFree(dimg_sin);
    return ok;
}

int iris_cuda_attention_tensor(iris_cuda_tensor_t out,
                               iris_cuda_tensor_t q, iris_cuda_tensor_t k, iris_cuda_tensor_t v,
                               int seq_q, int seq_k, int heads, int head_dim, float scale) {
    if (!out || !q || !k || !v) return 0;
    int hidden = heads * head_dim;
    size_t q_size = (size_t)seq_q * hidden;
    size_t k_size = (size_t)seq_k * hidden;

    if (iris_cuda_attention_fused_device(out->ptr, q->ptr, k->ptr, v->ptr,
                                         seq_q, seq_k, heads, head_dim, scale)) {
        return 1;
    }

    size_t head_size = (size_t)seq_q * head_dim;
    size_t kv_head_size = (size_t)seq_k * head_dim;
    size_t scores_size = (size_t)seq_q * seq_k;

    size_t total_gpu = q_size + k_size + k_size + q_size + ((size_t)heads * scores_size);
    float *gpu_buf = NULL;
    if (!cuda_check(cudaMalloc((void**)&gpu_buf, total_gpu * sizeof(float)), "attn tensor scratch")) {
        return 0;
    }

    float *dq = gpu_buf;
    float *dk = dq + q_size;
    float *dv = dk + k_size;
    float *do_t = dv + k_size;
    float *ds = do_t + q_size;

    if (!iris_cuda_transpose_shd_to_hsd(q->ptr, dq, seq_q, heads, head_dim) ||
        !iris_cuda_transpose_shd_to_hsd(k->ptr, dk, seq_k, heads, head_dim) ||
        !iris_cuda_transpose_shd_to_hsd(v->ptr, dv, seq_k, heads, head_dim)) {
        goto fail;
    }

    for (int h = 0; h < heads; h++) {
        float *qh = dq + (size_t)h * head_size;
        float *kh = dk + (size_t)h * kv_head_size;
        float *sh = ds + (size_t)h * scores_size;
        cuda_gemm(0, 1, seq_q, seq_k, head_dim, scale,
                  qh, head_dim, kh, head_dim, 0.0f, sh, seq_k);
    }
    iris_cuda_sync();
    if (!iris_cuda_softmax_inplace(ds, heads * seq_q, seq_k)) goto fail;

    for (int h = 0; h < heads; h++) {
        float *sh = ds + (size_t)h * scores_size;
        float *vh = dv + (size_t)h * kv_head_size;
        float *oh = do_t + (size_t)h * head_size;
        cuda_gemm(0, 0, seq_q, head_dim, seq_k, 1.0f,
                  sh, seq_k, vh, head_dim, 0.0f, oh, head_dim);
    }
    iris_cuda_sync();
    if (!iris_cuda_transpose_hsd_to_shd(do_t, out->ptr, seq_q, heads, head_dim)) goto fail;

    cudaFree(gpu_buf);
    return 1;

fail:
    if (gpu_buf) cudaFree(gpu_buf);
    return 0;
}

int iris_cuda_silu_mul_tensor(iris_cuda_tensor_t gate, iris_cuda_tensor_t up, int n) {
    if (!gate || !up) return 0;
    return iris_cuda_silu_mul_device(gate->ptr, up->ptr, n);
}

int iris_cuda_concat_attn_mlp(iris_cuda_tensor_t attn, iris_cuda_tensor_t mlp,
                              iris_cuda_tensor_t out, int seq, int hidden, int mlp_hidden) {
    if (!attn || !mlp || !out) return 0;
    return iris_cuda_concat_attn_mlp_device(attn->ptr, mlp->ptr, out->ptr,
                                            seq, hidden, mlp_hidden);
}

int iris_cuda_gated_add(iris_cuda_tensor_t hidden, const float *gate,
                        iris_cuda_tensor_t proj, int seq, int hidden_dim) {
    if (!hidden || !proj || !gate) return 0;
    float *dgate = cuda_upload(gate, hidden_dim);
    if (!dgate) return 0;
    int ok = iris_cuda_gated_add_device(hidden->ptr, dgate, proj->ptr, seq, hidden_dim);
    cudaFree(dgate);
    return ok;
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

    float *din = cuda_upload(in, (size_t)batch * in_ch * H * W);
    if (!din) return;

    float *dcol = NULL;
    if (!cuda_check(cudaMalloc((void**)&dcol, col_n * sizeof(float)), "im2col malloc")) {
        cudaFree(din);
        return;
    }

    float *dw = cuda_upload(weight, (size_t)out_ch * K);
    if (!dw) { cudaFree(din); cudaFree(dcol); return; }

    float *dout = NULL;
    cudaMalloc((void**)&dout, (size_t)batch * out_ch * tile_pixels * sizeof(float));
    if (!dout) { cudaFree(din); cudaFree(dcol); cudaFree(dw); return; }

    for (int b = 0; b < batch; b++) {
        float *din_b = din + (size_t)b * in_ch * H * W;
        float *dout_b = dout + b * out_ch * tile_pixels;
        if (!iris_cuda_im2col(din_b, dcol, in_ch, H, W, kH, kW, stride, padding, outH, outW)) {
            cudaFree(din); cudaFree(dcol); cudaFree(dw); cudaFree(dout);
            return;
        }
        cuda_gemm(0, 0, out_ch, tile_pixels, K,
                  1.0f, dw, K, dcol, tile_pixels, 0.0f, dout_b, tile_pixels);
    }

    if (bias) {
        float *db = cuda_upload(bias, out_ch);
        if (db) {
            iris_cuda_conv2d_add_bias(dout, db, batch, out_ch, tile_pixels);
            cudaFree(db);
        }
    }

    cuda_download(out, dout, (size_t)batch * out_ch * tile_pixels);

    cudaFree(din); cudaFree(dcol); cudaFree(dw); cudaFree(dout);
}

/* --------------------------------------------------------------- */
/*  Multi-head attention: handles the [seq, heads*head_dim] layout  */
/* --------------------------------------------------------------- */

int iris_cuda_attention(float *out,
                        const float *Q, const float *K, const float *V,
                        int seq_q, int seq_k, int heads, int head_dim,
                        float scale) {
    int hidden = heads * head_dim;

    size_t q_size = (size_t)seq_q * hidden;
    size_t k_size = (size_t)seq_k * hidden;
    size_t head_size = (size_t)seq_q * head_dim;
    size_t kv_head_size = (size_t)seq_k * head_dim;
    size_t scores_size = (size_t)seq_q * seq_k;

    /* Allocate all GPU buffers in one chunk to avoid WSL driver heap issue */
    size_t total_gpu = (q_size * 4) + (k_size * 4) + ((size_t)heads * scores_size);
    float *gpu_buf = NULL;
    cudaError_t ce = cudaMalloc((void**)&gpu_buf, total_gpu * sizeof(float));
    if (ce != cudaSuccess || !gpu_buf) {
        return 0;
    }

    float *dq_shd = gpu_buf;
    float *dk_shd = dq_shd + q_size;
    float *dv_shd = dk_shd + k_size;
    float *do_shd = dv_shd + k_size;
    float *dq = do_shd + q_size;
    float *dk = dq + q_size;
    float *dv = dk + k_size;
    float *do_t = dv + k_size;
    float *ds = do_t + q_size;

    cudaMemcpy(dq_shd, Q, q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk_shd, K, k_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv_shd, V, k_size * sizeof(float), cudaMemcpyHostToDevice);

    if (getenv("IRIS_CUDA_FUSED_ATTN") &&
        iris_cuda_attention_fused_device(do_shd, dq_shd, dk_shd, dv_shd,
                                         seq_q, seq_k, heads, head_dim, scale)) {
        iris_cuda_sync();
        cuda_download(out, do_shd, q_size);
        cudaFree(gpu_buf);
        return 1;
    }

    if (!iris_cuda_transpose_shd_to_hsd(dq_shd, dq, seq_q, heads, head_dim) ||
        !iris_cuda_transpose_shd_to_hsd(dk_shd, dk, seq_k, heads, head_dim) ||
        !iris_cuda_transpose_shd_to_hsd(dv_shd, dv, seq_k, heads, head_dim)) {
        cudaFree(gpu_buf);
        return 0;
    }

    /* Per-head cuBLAS calls (avoids cublasSgemmStridedBatched bug in WSL driver) */
    for (int h = 0; h < heads; h++) {
        float *qh = dq + h * head_size;
        float *kh = dk + h * kv_head_size;
        float *sh = ds + h * scores_size;

        /* scores_h[seq_q, seq_k] = Q_h[seq_q, head_dim] @ K_h^T[head_dim, seq_k] */
        cuda_gemm(0, 1, seq_q, seq_k, head_dim, scale,
                  qh, head_dim, kh, head_dim, 0.0f, sh, seq_k);
    }

    iris_cuda_sync();

    if (!iris_cuda_softmax_inplace(ds, heads * seq_q, seq_k)) {
        cudaFree(gpu_buf);
        return 0;
    }

    /* out_h[seq_q, head_dim] = scores_h[seq_q, seq_k] @ V_h[seq_k, head_dim] */
    for (int h = 0; h < heads; h++) {
        float *sh = ds + h * scores_size;
        float *vh = dv + h * kv_head_size;
        float *oh = do_t + h * head_size;

        cuda_gemm(0, 0, seq_q, head_dim, seq_k, 1.0f,
                  sh, seq_k, vh, head_dim, 0.0f, oh, head_dim);
    }

    iris_cuda_sync();

    if (!iris_cuda_transpose_hsd_to_shd(do_t, do_shd, seq_q, heads, head_dim)) {
        cudaFree(gpu_buf);
        return 0;
    }
    cuda_download(out, do_shd, q_size);

    cudaFree(gpu_buf);
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
