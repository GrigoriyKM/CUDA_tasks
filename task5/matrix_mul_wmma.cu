#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

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

__global__ void matrixMulTensorCore(half *A, half *B, float *C, int M, int N, int K)
{
    // Define the fragment variables
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the accumulator fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the matrices into the fragments
    wmma::load_matrix_sync(a_frag, A + (blockIdx.x * 16) * K + (threadIdx.x * 16), K);
    wmma::load_matrix_sync(b_frag, B + (blockIdx.y * 16) * N + (threadIdx.y * 16), N);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result back to global memory
    wmma::store_matrix_sync(C + (blockIdx.x * 16) * N + (blockIdx.y * 16), c_frag, N, wmma::mem_row_major);
}
int main()
{
    int M = 3000;
    int N = 4500;
    int K = 6000;

    M = toMultiple(M, 16);
    N = toMultiple(N, 16);
    K = toMultiple(K, 16);

    half *A, *B;
    float *C;

    // Allocate memory on the device
    cudaMalloc((void **)&A, M * K * sizeof(half));
    cudaMalloc((void **)&B, K * N * sizeof(half));
    cudaMalloc((void **)&C, M * N * sizeof(float));

    // Initialize host data
    half h_A[M * K];
    half h_B[K * N];
    float h_C[M * N];

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<half>(i);
    }

    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = static_cast<half>(i);
    }

    // Copy data to device
    cudaMemcpy(A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(1, 1);
    dim3 gridSize((M + 15) / 16, (N + 15) / 16);

    matrixMulTensorCore<<<gridSize, blockSize>>>(A, B, C, M, N, K);

    // Copy result back to host
    cudaMemcpy(h_C, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
