#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <time.h>

const int N = 1000000000; // 10^9

// Ядро для инициализации массива
__global__ void initializeArray(float *arr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        arr[i] = __sinf((i % 360) * M_PI / 180);
    }
}

double calcError(float *hostArr, int arraySize)
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
    // Время начала выполнения
    clock_t start = clock();
    int device = 1;
    cudaSetDevice(device);
    // Выделение памяти на GPU для массива
    float *d_arr;
    checkCudaError(cudaMalloc(&d_arr, N * sizeof(float)));

    dim3 blockSize = dim3(256);
    dim3 numBlocks = dim3((N + blockSize.x - 1) / blockSize.x);
    // Запуск ядра для инициализации массива
    initializeArray<<<numBlocks, blockSize>>>(d_arr);

    // Выделение памяти на CPU для массива
    float *h_arr = (float *)malloc(N * sizeof(float));

    // Копирование массива с GPU на CPU
    checkCudaError(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaDeviceSynchronize());

    // Расчет ошибки
    printf("Ошибка (__sinf) = %0.10f \n", calcError(h_arr, N));

    // Освобождение памяти на GPU
    cudaFree(d_arr);

    // Освобождение памяти на CPU
    free(h_arr);

    // Время окончания выполнения
    clock_t end = clock();

    // Вывод времени выполнения
    printf("Время выполнения: %0.5f секунд \n", (end - start) / CLOCKS_PER_SEC);

    return 0;
};