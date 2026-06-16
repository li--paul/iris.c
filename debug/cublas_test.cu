#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    int M = 15360, N = 1280, K = 3072;

    float *A, *B, *C;
    cudaMalloc(&A, (size_t)M * K * sizeof(float));
    cudaMalloc(&B, (size_t)K * N * sizeof(float));
    cudaMalloc(&C, (size_t)M * N * sizeof(float));

    cudaMemset(A, 0, (size_t)M * K * sizeof(float));
    cudaMemset(B, 0, (size_t)K * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, A, K, B, K, &beta, C, M);
    printf("sgemm(%dx%dx%d): status=%d\n", M, N, K, (int)s);

    cublasDestroy(handle);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return s == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
