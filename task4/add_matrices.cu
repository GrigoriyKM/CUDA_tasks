#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
// размер блока
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE float

// Ядро
__global__ void matrixAdd(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int cols)
{
    // Вычисление индекса элемента матрицы на GPU
    int ind = cols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = A[ind] + B[ind];
}
// Функция вычисление числа, которое больше
// числа а и кратное числу b
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
    // start, stop - for Kernel time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // количество строк и столбцов матрицы
    int rows = 1000;
    int cols = 2000;

    rows = toMultiple(rows, BLOCK_SIZE);
    printf("rows = %d\n", rows);

    cols = toMultiple(cols, BLOCK_SIZE);
    printf("cols = %d\n", cols);

    size_t size = rows * cols * sizeof(BASE_TYPE);

    BASE_TYPE *h_A = (BASE_TYPE *)malloc(size);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(size);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(size);

    for (int i = 0; i < rows * cols; ++i)
    {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;

        BASE_TYPE *d_A = NULL;
        cudaMalloc((void **)&d_A, size);

        BASE_TYPE *d_B = NULL;
        cudaMalloc((void **)&d_B, size);

        BASE_TYPE *d_C = NULL;
        cudaMalloc((void **)&d_C, size);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocksPerGrid = dim3(cols / BLOCK_SIZE, rows / BLOCK_SIZE);

        cudaEventRecord(start, 0);

        matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, cols);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float KernelTime;
        cudaEventElapsedTime(&KernelTime, start, stop);
        printf("KernelTime: %.2f milliseconds\n", KernelTime);

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        printf("Test STARTED\n");
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                BASE_TYPE sum;
                sum = h_A[i * cols + j] + h_B[i * cols + j];

                if (fabs(h_C[i * cols + j] - sum) > 1e-3)
                {
                    fprintf(stderr, "Result verification failed at element [%d, %d]!\n", i, j);
                    printf("sum = %f, h_C[i * cols + j] = %f\n", sum, h_C[i * cols + j]);
                    exit(EXIT_FAILURE);
                }
            }
        }
        printf("Test PASSED\n");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 0;
    }
}