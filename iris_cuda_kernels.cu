#include "iris_cuda.h"

#include <cuda_runtime.h>

__global__ static void transpose_shd_to_hsd_kernel(const float *src, float *dst,
                                                   int seq, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int s = idx / (head_dim * heads);
    dst[(h * seq + s) * head_dim + d] = src[(s * heads + h) * head_dim + d];
}

__global__ static void transpose_hsd_to_shd_kernel(const float *src, float *dst,
                                                   int seq, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int s = idx / (head_dim * heads);
    dst[(s * heads + h) * head_dim + d] = src[(h * seq + s) * head_dim + d];
}

__global__ static void softmax_rows_kernel(float *x, int rows, int cols) {
    extern __shared__ float shared[];
    float *smax = shared;
    float *ssum = shared + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float *row_ptr = x + (size_t)row * cols;

    float max_val = -3.402823466e+38f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = row_ptr[i];
        max_val = (v > max_val) ? v : max_val;
    }
    smax[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && smax[tid + stride] > smax[tid]) smax[tid] = smax[tid + stride];
        __syncthreads();
    }

    max_val = smax[0];
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float e = __expf(row_ptr[i] - max_val);
        row_ptr[i] = e;
        sum += e;
    }
    ssum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }

    float inv_sum = 1.0f / ssum[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        row_ptr[i] *= inv_sum;
    }
}

__global__ static void add_bias_rows_kernel(float *x, const float *bias, int total, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] += bias[idx % cols];
}

__global__ static void silu_mul_kernel(float *gate, const float *up, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = gate[idx];
    gate[idx] = (v / (1.0f + __expf(-v))) * up[idx];
}

__global__ static void rms_norm_rows_kernel(float *out, const float *x, const float *weight,
                                            int rows, int hidden, float eps) {
    extern __shared__ float ss[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *x_row = x + (size_t)row * hidden;
    float *out_row = out + (size_t)row * hidden;

    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    ss[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ss[tid] += ss[tid + stride];
        __syncthreads();
    }

    float inv = rsqrtf(ss[0] / hidden + eps);
    for (int i = tid; i < hidden; i += blockDim.x) {
        out_row[i] = x_row[i] * inv * weight[i];
    }
}

__global__ static void adaln_norm_rows_kernel(float *out, const float *x,
                                              const float *shift, const float *scale,
                                              int rows, int hidden, float eps) {
    extern __shared__ float ss[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *x_row = x + (size_t)row * hidden;
    float *out_row = out + (size_t)row * hidden;

    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) sum += x_row[i];
    ss[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ss[tid] += ss[tid + stride];
        __syncthreads();
    }
    float mean = ss[0] / hidden;

    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float d = x_row[i] - mean;
        var_sum += d * d;
    }
    ss[tid] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ss[tid] += ss[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(ss[0] / hidden + eps);

    for (int i = tid; i < hidden; i += blockDim.x) {
        float norm = (x_row[i] - mean) * inv_std;
        out_row[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

__global__ static void rope_split_half_kernel(float *x, const float *freqs,
                                              int batch, int seq, int heads, int head_dim) {
    int half_dim = head_dim / 2;
    int total = batch * seq * heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % half_dim;
    int h = (idx / half_dim) % heads;
    int s = (idx / (half_dim * heads)) % seq;
    int b = idx / (half_dim * heads * seq);
    float *vec = x + (((size_t)b * seq + s) * heads + h) * head_dim;
    const float *freq = freqs + ((size_t)s * half_dim + d) * 2;

    float c = freq[0];
    float sn = freq[1];
    float x0 = vec[d];
    float x1 = vec[d + half_dim];
    vec[d] = x0 * c - x1 * sn;
    vec[d + half_dim] = x0 * sn + x1 * c;
}

__global__ static void im2col_kernel(const float *in, float *col,
                                     int in_ch, int H, int W, int kH, int kW,
                                     int stride, int padding, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_pixels = outH * outW;
    int K = in_ch * kH * kW;
    int total = K * tile_pixels;
    if (idx >= total) return;

    int pixel = idx % tile_pixels;
    int col_row = idx / tile_pixels;
    int ow = pixel % outW;
    int oh = pixel / outW;
    int kw = col_row % kW;
    int kh = (col_row / kW) % kH;
    int ic = col_row / (kH * kW);
    int ih = oh * stride - padding + kh;
    int iw = ow * stride - padding + kw;

    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        v = in[(ic * H + ih) * W + iw];
    }
    col[idx] = v;
}

__global__ static void conv2d_add_bias_kernel(float *out, const float *bias,
                                              int total, int out_ch, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int oc = (idx / pixels) % out_ch;
    out[idx] += bias[oc];
}

__global__ static void split_qkv_mlp_kernel(const float *fused,
                                            float *q, float *k, float *v,
                                            float *gate, float *up,
                                            int seq, int hidden, int mlp_hidden) {
    int fused_dim = hidden * 3 + mlp_hidden * 2;
    int total_hidden = seq * hidden;
    int total_mlp = seq * mlp_hidden;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_hidden) {
        int s = idx / hidden;
        int d = idx % hidden;
        const float *row = fused + (size_t)s * fused_dim;
        q[idx] = row[d];
        k[idx] = row[hidden + d];
        v[idx] = row[hidden * 2 + d];
    }

    if (idx < total_mlp) {
        int s = idx / mlp_hidden;
        int d = idx % mlp_hidden;
        const float *row = fused + (size_t)s * fused_dim;
        gate[idx] = row[hidden * 3 + d];
        up[idx] = row[hidden * 3 + mlp_hidden + d];
    }
}

__global__ static void qk_rms_norm_kernel(float *q, float *k,
                                          const float *q_weight, const float *k_weight,
                                          int seq, int heads, int head_dim, float eps) {
    extern __shared__ float ss[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_rows = seq * heads;
    if (row >= total_rows) return;

    float *qh = q + (size_t)row * head_dim;
    float *kh = k + (size_t)row * head_dim;

    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float qv = qh[d];
        float kv = kh[d];
        sum_q += qv * qv;
        sum_k += kv * kv;
    }
    ss[tid] = sum_q;
    ss[blockDim.x + tid] = sum_k;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ss[tid] += ss[tid + stride];
            ss[blockDim.x + tid] += ss[blockDim.x + tid + stride];
        }
        __syncthreads();
    }

    float inv_q = rsqrtf(ss[0] / head_dim + eps);
    float inv_k = rsqrtf(ss[blockDim.x] / head_dim + eps);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        qh[d] = qh[d] * inv_q * q_weight[d];
        kh[d] = kh[d] * inv_k * k_weight[d];
    }
}

__global__ static void rope_unified_kernel(float *q, float *k,
                                           const float *txt_cos, const float *txt_sin,
                                           const float *img_cos, const float *img_sin,
                                           int seq, int img_offset, int heads, int head_dim) {
    int pairs = head_dim / 2;
    int total = seq * heads * pairs;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int pair = idx % pairs;
    int h = (idx / pairs) % heads;
    int s = idx / (pairs * heads);
    int d = pair * 2;
    const float *cos_row;
    const float *sin_row;
    if (s < img_offset) {
        cos_row = txt_cos + (size_t)s * head_dim;
        sin_row = txt_sin + (size_t)s * head_dim;
    } else {
        int img_s = s - img_offset;
        cos_row = img_cos + (size_t)img_s * head_dim;
        sin_row = img_sin + (size_t)img_s * head_dim;
    }

    float c = cos_row[d];
    float sn = sin_row[d];
    float *qv = q + ((size_t)s * heads + h) * head_dim + d;
    float *kv = k + ((size_t)s * heads + h) * head_dim + d;

    float q0 = qv[0];
    float q1 = qv[1];
    qv[0] = q0 * c - q1 * sn;
    qv[1] = q1 * c + q0 * sn;

    float k0 = kv[0];
    float k1 = kv[1];
    kv[0] = k0 * c - k1 * sn;
    kv[1] = k1 * c + k0 * sn;
}

__global__ static void concat_attn_mlp_kernel(const float *attn, const float *mlp,
                                              float *out, int seq, int hidden, int mlp_hidden) {
    int out_dim = hidden + mlp_hidden;
    int total = seq * out_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s = idx / out_dim;
    int d = idx % out_dim;
    if (d < hidden) out[idx] = attn[(size_t)s * hidden + d];
    else out[idx] = mlp[(size_t)s * mlp_hidden + (d - hidden)];
}

__global__ static void gated_add_kernel(float *hidden_buf, const float *gate,
                                        const float *proj, int total, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    hidden_buf[idx] += gate[idx % hidden] * proj[idx];
}

__global__ static void attention_fused_kernel(float *out,
                                              const float *Q, const float *K, const float *V,
                                              int seq_q, int seq_k, int heads, int head_dim,
                                              float scale) {
    extern __shared__ float shared[];
    float *scratch = shared;
    int h = blockIdx.x;
    int q_idx = blockIdx.y;
    int tid = threadIdx.x;
    int hidden = heads * head_dim;
    if (h >= heads || q_idx >= seq_q) return;

    const float *q = Q + (size_t)q_idx * hidden + h * head_dim;

    float row_max = -3.402823466e+38f;
    for (int k_idx = 0; k_idx < seq_k; k_idx++) {
        const float *k = K + (size_t)k_idx * hidden + h * head_dim;
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) dot += q[d] * k[d];
        scratch[tid] = dot;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float score = scratch[0] * scale;
            row_max = (score > row_max) ? score : row_max;
        }
        __syncthreads();
    }

    if (tid == 0) scratch[0] = row_max;
    __syncthreads();
    float max_shared = scratch[0];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        float denom = 0.0f;
        for (int k_idx = 0; k_idx < seq_k; k_idx++) {
            const float *k = K + (size_t)k_idx * hidden + h * head_dim;
            float dot = 0.0f;
            for (int rd = 0; rd < head_dim; rd++) dot += q[rd] * k[rd];
            float w = __expf(dot * scale - max_shared);
            denom += w;
            acc += w * V[(size_t)k_idx * hidden + h * head_dim + d];
        }
        out[(size_t)q_idx * hidden + h * head_dim + d] = acc / denom;
    }
}

extern "C" int iris_cuda_split_qkv_mlp_device(const float *d_fused,
                                               float *d_q, float *d_k, float *d_v,
                                               float *d_gate, float *d_up,
                                               int seq, int hidden, int mlp_hidden) {
    int total_hidden = seq * hidden;
    int total_mlp = seq * mlp_hidden;
    int total = total_hidden > total_mlp ? total_hidden : total_mlp;
    if (!d_fused || !d_q || !d_k || !d_v || !d_gate || !d_up || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    split_qkv_mlp_kernel<<<grid, block>>>(d_fused, d_q, d_k, d_v, d_gate, d_up,
                                          seq, hidden, mlp_hidden);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_qk_rms_norm_device(float *d_q, float *d_k,
                                             const float *d_q_weight, const float *d_k_weight,
                                             int seq, int heads, int head_dim, float eps) {
    if (!d_q || !d_k || !d_q_weight || !d_k_weight || seq <= 0 || heads <= 0 || head_dim <= 0) return 0;
    int threads = 256;
    if (head_dim < threads) {
        threads = 1;
        while (threads < head_dim) threads <<= 1;
        if (threads < 32) threads = 32;
    }
    qk_rms_norm_kernel<<<seq * heads, threads, (size_t)threads * 2 * sizeof(float)>>>(
        d_q, d_k, d_q_weight, d_k_weight, seq, heads, head_dim, eps);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_rope_unified_device(float *d_q, float *d_k,
                                              const float *d_txt_cos, const float *d_txt_sin,
                                              const float *d_img_cos, const float *d_img_sin,
                                              int seq, int img_offset, int heads, int head_dim) {
    int total = seq * heads * (head_dim / 2);
    if (!d_q || !d_k || !d_txt_cos || !d_txt_sin || !d_img_cos || !d_img_sin ||
        total <= 0 || (head_dim & 1)) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_unified_kernel<<<grid, block>>>(d_q, d_k, d_txt_cos, d_txt_sin,
                                         d_img_cos, d_img_sin,
                                         seq, img_offset, heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_concat_attn_mlp_device(const float *d_attn, const float *d_mlp,
                                                 float *d_out, int seq, int hidden, int mlp_hidden) {
    int total = seq * (hidden + mlp_hidden);
    if (!d_attn || !d_mlp || !d_out || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    concat_attn_mlp_kernel<<<grid, block>>>(d_attn, d_mlp, d_out, seq, hidden, mlp_hidden);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_gated_add_device(float *d_hidden, const float *d_gate,
                                           const float *d_proj, int seq, int hidden) {
    int total = seq * hidden;
    if (!d_hidden || !d_gate || !d_proj || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    gated_add_kernel<<<grid, block>>>(d_hidden, d_gate, d_proj, total, hidden);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_softmax_inplace(float *d_x, int rows, int cols) {
    if (!d_x || rows <= 0 || cols <= 0) return 0;

    int threads = 256;
    if (cols < threads) {
        threads = 1;
        while (threads < cols) threads <<= 1;
        if (threads < 32) threads = 32;
    }

    size_t shared_bytes = (size_t)threads * 2 * sizeof(float);
    softmax_rows_kernel<<<rows, threads, shared_bytes>>>(d_x, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_transpose_shd_to_hsd(const float *d_src, float *d_dst,
                                               int seq, int heads, int head_dim) {
    int total = seq * heads * head_dim;
    if (!d_src || !d_dst || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_shd_to_hsd_kernel<<<grid, block>>>(d_src, d_dst, seq, heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_transpose_hsd_to_shd(const float *d_src, float *d_dst,
                                               int seq, int heads, int head_dim) {
    int total = seq * heads * head_dim;
    if (!d_src || !d_dst || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_hsd_to_shd_kernel<<<grid, block>>>(d_src, d_dst, seq, heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_add_bias_rows(float *d_x, const float *d_bias, int rows, int cols) {
    int total = rows * cols;
    if (!d_x || !d_bias || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_rows_kernel<<<grid, block>>>(d_x, d_bias, total, cols);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_silu_mul_device(float *d_gate, const float *d_up, int n) {
    if (!d_gate || !d_up || n <= 0) return 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block>>>(d_gate, d_up, n);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_rms_norm_device(float *d_out, const float *d_x, const float *d_weight,
                                          int rows, int hidden, float eps) {
    if (!d_out || !d_x || !d_weight || rows <= 0 || hidden <= 0) return 0;
    int threads = 256;
    if (hidden < threads) {
        threads = 1;
        while (threads < hidden) threads <<= 1;
        if (threads < 32) threads = 32;
    }
    rms_norm_rows_kernel<<<rows, threads, (size_t)threads * sizeof(float)>>>(d_out, d_x, d_weight,
                                                                             rows, hidden, eps);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_adaln_norm_device(float *d_out, const float *d_x,
                                            const float *d_shift, const float *d_scale,
                                            int rows, int hidden, float eps) {
    if (!d_out || !d_x || !d_shift || !d_scale || rows <= 0 || hidden <= 0) return 0;
    int threads = 256;
    if (hidden < threads) {
        threads = 1;
        while (threads < hidden) threads <<= 1;
        if (threads < 32) threads = 32;
    }
    adaln_norm_rows_kernel<<<rows, threads, (size_t)threads * sizeof(float)>>>(d_out, d_x,
                                                                               d_shift, d_scale,
                                                                               rows, hidden, eps);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_apply_rope_device(float *d_x, const float *d_freqs,
                                            int batch, int seq, int heads, int head_dim) {
    int total = batch * seq * heads * (head_dim / 2);
    if (!d_x || !d_freqs || total <= 0 || (head_dim & 1)) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_split_half_kernel<<<grid, block>>>(d_x, d_freqs, batch, seq, heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_im2col(const float *d_in, float *d_col,
                                 int in_ch, int H, int W, int kH, int kW,
                                 int stride, int padding, int outH, int outW) {
    int total = in_ch * kH * kW * outH * outW;
    if (!d_in || !d_col || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    im2col_kernel<<<grid, block>>>(d_in, d_col, in_ch, H, W, kH, kW, stride, padding, outH, outW);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_conv2d_add_bias(float *d_out, const float *d_bias,
                                          int batch, int out_ch, int pixels) {
    int total = batch * out_ch * pixels;
    if (!d_out || !d_bias || total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_add_bias_kernel<<<grid, block>>>(d_out, d_bias, total, out_ch, pixels);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int iris_cuda_attention_fused_device(float *d_out,
                                                 const float *d_Q, const float *d_K, const float *d_V,
                                                 int seq_q, int seq_k, int heads, int head_dim,
                                                 float scale) {
    if (!d_out || !d_Q || !d_K || !d_V || seq_q <= 0 || seq_k <= 0 || heads <= 0 || head_dim <= 0) return 0;
    if (head_dim > 256 || seq_k > 512) return 0;
    dim3 grid(heads, seq_q);
    int threads = 256;
    size_t shared_bytes = (size_t)threads * sizeof(float);
    attention_fused_kernel<<<grid, threads, shared_bytes>>>(d_out, d_Q, d_K, d_V,
                                                            seq_q, seq_k, heads, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}
