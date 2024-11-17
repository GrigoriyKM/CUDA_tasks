#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

int main()
{
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    int M = 3000;
    int N = 4500;
    int K = 6000;

    M = toMultiple(M, 16);
    N = toMultiple(N, 16);
    K = toMultiple(K, 16);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    float *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // Initialize host data
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<float>(i);
    }

    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = static_cast<float>(i);
    }

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result (for small matrices)
    if (M <= 16 && N <= 16)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                std::cout << h_C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    return 0;
}
