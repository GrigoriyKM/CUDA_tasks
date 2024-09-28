#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <time.h>

const int N = 1000000000; // 10^9

// Ядро для инициализации массива
__global__ void initializeArray(double *arr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        arr[i] = __sinf((i % 360) * M_PI / 180);
    }
}

double calcError(double *hostArr, int arraySize)
{
    double err = 0;
    for (int i = 0; i < arraySize; i++)
    {
        err += abs(sin((i % 360) * M_PI / 180) - hostArr[i]);
    }
    return err / arraySize;
}

void checkCudaError(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

int main()
{

    int device = 1;
    cudaSetDevice(device);
    // Выделение памяти на GPU для массива
    double *d_arr;
    checkCudaError(cudaMalloc(&d_arr, N * sizeof(double)));

    dim3 blockSize = dim3(256);
    dim3 numBlocks = dim3((N + blockSize.x - 1) / blockSize.x);

    // Время начала выполнения
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Запуск ядра для инициализации массива
    initializeArray<<<numBlocks, blockSize>>>(d_arr);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Выделение памяти на CPU для массива
    double *h_arr = (double *)malloc(N * sizeof(double));

    // Копирование массива с GPU на CPU
    checkCudaError(cudaMemcpy(h_arr, d_arr, N * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaError(cudaDeviceSynchronize());

    // Расчет ошибки
    printf("Ошибка (__sinf) = %0.10f \n", calcError(h_arr, N));

    // Вывод времени выполнения
    printf("Время выполнения: %f мсек \n", milliseconds);

    // Освобождение памяти на GPU
    cudaFree(d_arr);

    // Освобождение памяти на CPU
    free(h_arr);

    return 0;
};