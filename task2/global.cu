#include <iostream>
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

float sobelKernelX[3 * 3] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1};

float sobelKernelY[3 * 3] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1};

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
                            unsigned int width, unsigned int height, float *kernel)
{
    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    int x_i = x_o - FILTER_SIZE / 2;
    int y_i = y_o - FILTER_SIZE / 2;

    int sum = 0;
    if ((threadIdx.x < TILE_SIZE) && (threadIdx.y < TILE_SIZE))
    {

        for (int r = 0; r < FILTER_SIZE; ++r)
        {
            for (int c = 0; c < FILTER_SIZE; ++c)
            {
                if (x_i + c >= 0 && x_i + c < width && y_i + r >= 0 && y_i + r < height)
                {
                    float k_value = kernel[r * FILTER_SIZE + c];
                    sum += in[(y_i + r) * width + x_i + c] * k_value;
                }
            }
        }
        // sum = sum / (FILTER_SIZE * FILTER_SIZE);
        // write into the output
        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;
    }
}

cudaTextureObject_t createTexture(cudaArray_t array)
{
    cudaTextureObject_t textureObject = 0;

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    CUDA_CHECK_RETURN(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

    return textureObject;
}

int main(int, char **)
{
    std::cout << "Используемая память: global memory" << std::endl;

    cv::Mat img = cv::imread("Lenna.png", cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
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

    // результат фильтрации на устройстве
    unsigned char *d_r_n, *d_g_n, *d_b_n;
    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size));

    unsigned char *d_r, *d_g, *d_b;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, size));

    CUDA_CHECK_RETURN(cudaMemcpy(d_r, channels[2].data, size, cudaMemcpyHostToDevice)); // R
    CUDA_CHECK_RETURN(cudaMemcpy(d_g, channels[1].data, size, cudaMemcpyHostToDevice)); // G
    CUDA_CHECK_RETURN(cudaMemcpy(d_b, channels[0].data, size, cudaMemcpyHostToDevice)); // B

    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    // ядро гаусса
    float h_kernel[FILTER_SIZE * FILTER_SIZE];
    createGaussianKernel(h_kernel, FILTER_SIZE, SIGMA);
    float *d_kernel;
    cudaMalloc(&d_kernel, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    applyFilter<<<grid_size, blockSize>>>(d_r_n, d_r, width, height, d_kernel);
    applyFilter<<<grid_size, blockSize>>>(d_g_n, d_g, width, height, d_kernel);
    applyFilter<<<grid_size, blockSize>>>(d_b_n, d_b, width, height, d_kernel);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));

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
    cudaFree(d_r_n);
    cudaFree(d_g_n);
    cudaFree(d_b_n);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);

    std::cout << "Результат фильтрации: 'filtred_image.png'!" << std::endl;
    std::cout << "Время выполнения: " << milliseconds << " мсек" << std::endl;

    return 0;
}