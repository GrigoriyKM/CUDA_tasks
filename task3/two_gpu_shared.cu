#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <array>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE (32u)
#define FILTER_SIZE (9u)
#define TILE_SIZE (23u) // BLOCK_SIZE - 2( FILTER_SIZE/2)
#define SIGMA 2.0f

#define CUDA_CHECK_RETURN(value)                                  \
    {                                                             \
        cudaError_t err = value;                                  \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "Error %s at line %d in file %s\n",   \
                    cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(-1);                                             \
        }                                                         \
    }

void createGaussianKernel(float *kernel, int k_size, float sigma)
{
    float sum = 0.0f;

    for (int y = -k_size / 2; y <= k_size / 2; y++)
    {
        for (int x = -k_size / 2; x <= k_size / 2; x++)
        {
            kernel[(y + k_size / 2) * k_size + (x + k_size / 2)] =
                (1.0f / (2.0f * M_PI * sigma * sigma)) *
                expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            sum += kernel[(y + k_size / 2) * k_size + (x + k_size / 2)];
        }
    }

    for (int i = 0; i < k_size * k_size; i++)
    {
        kernel[i] /= sum;
    }
}

__global__ void applyFilter(unsigned char *out, unsigned char *in,
                            unsigned int width, unsigned int height)
{
    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    int x_i = x_o - FILTER_SIZE / 2;
    int y_i = y_o - FILTER_SIZE / 2;

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height))
    {
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * width + x_i];
    }
    else
    {
        sBuffer[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    int sum = 0;
    if ((threadIdx.x < TILE_SIZE) && (threadIdx.y < TILE_SIZE))
    {

        for (int r = 0; r < FILTER_SIZE; ++r)
        {
            for (int c = 0; c < FILTER_SIZE; ++c)
            {
                // float k_value = kernel[r * FILTER_SIZE + c];
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c];
            }
        }
        sum = sum / (FILTER_SIZE * FILTER_SIZE);
        // write into the output
        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;
    }
}

void filterImageWithGPUs(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int startY, int endY, int gpuId)
{
    unsigned char *d_input, *d_output;

    // Adjust the startY and endY to include overlap (e.g. 1 pixel above and below)
    int overlap = FILTER_SIZE / 2; // Change this for larger overlap as needed
    int adjustedStartY = max(startY - overlap, 0);
    int adjustedEndY = min(endY + overlap, height);

    // Allocate memory on the device
    CUDA_CHECK_RETURN(cudaSetDevice(gpuId));
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    CUDA_CHECK_RETURN(cudaEventRecord(start));

    int totalHeight = adjustedEndY - adjustedStartY; // New height with overlap
    unsigned int d_size = width * totalHeight * sizeof(unsigned char);

    CUDA_CHECK_RETURN(cudaMalloc(&d_input, d_size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, d_size));

    // Copy adjusted input image section to GPU (includes overlap)
    CUDA_CHECK_RETURN(cudaMemcpy(d_input,
                                 inputImage + adjustedStartY * width, totalHeight * width * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Launch the Gaussian blur kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (totalHeight + TILE_SIZE - 1) / TILE_SIZE);

    applyFilter<<<gridSize, blockSize>>>(d_output, d_input, width, totalHeight);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize())

    // Copy the output back to the host for the valid output section
    CUDA_CHECK_RETURN(cudaMemcpy(outputImage + (startY * width),
                                 d_output + (startY - adjustedStartY) * width, (endY - startY) * width * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Время обработки половины канала: " << milliseconds << " мсек" << std::endl;
    // Free device memory
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));
}

int main()
{
    std::cout << "Используемая память: shared memory" << std::endl;

    cv::Mat img = cv::imread("fox.png", cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Ошибка загрузки изображения!" << std::endl;
        return -1;
    }

    unsigned int width = img.cols;
    unsigned int height = img.rows;
    unsigned int size = width * height * sizeof(unsigned char);

    // результат фильтрации на хосте
    unsigned char *h_r_n = (unsigned char *)malloc(size);
    unsigned char *h_g_n = (unsigned char *)malloc(size);
    unsigned char *h_b_n = (unsigned char *)malloc(size);

    cv::Mat channels[3];
    cv::split(img, channels);

    int halfHeight = height / 2;

#pragma omp parallel sections
    {
#pragma omp section

        {
            // GPU 0
            std::cout << "Обработка верхней части изображения на GPU 0..." << std::endl;
            filterImageWithGPUs(channels[2].data, h_r_n, width, height, 0, halfHeight, 0);
            filterImageWithGPUs(channels[1].data, h_g_n, width, height, 0, halfHeight, 0);
            filterImageWithGPUs(channels[0].data, h_b_n, width, height, 0, halfHeight, 0);
            std::cout << "finished на GPU 0..." << std::endl;
        }

#pragma omp section
        {
            // GPU 1
            std::cout << "Обработка верхней части изображения на GPU 1..." << std::endl;
            filterImageWithGPUs(channels[2].data, h_r_n, width, height, halfHeight, height, 1);
            filterImageWithGPUs(channels[1].data, h_g_n, width, height, halfHeight, height, 1);
            filterImageWithGPUs(channels[0].data, h_b_n, width, height, halfHeight, height, 1);
            std::cout << "finished на GPU 1..." << std::endl;
        }
    }

    cv::Mat output_img(height, width, CV_8UC3);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            output_img.at<cv::Vec3b>(i, j)[0] = h_b_n[i * width + j]; // B
            output_img.at<cv::Vec3b>(i, j)[1] = h_g_n[i * width + j]; // G
            output_img.at<cv::Vec3b>(i, j)[2] = h_r_n[i * width + j]; // R
        }
    }

    cv::imwrite("filtred_image.png", output_img);
    free(h_r_n);
    free(h_g_n);
    free(h_b_n);

    return 0;
}
