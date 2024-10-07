#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define BLOCK_SIZE 32
#define FILTER_RADIUS 6
#define KERNEL_SIZE 13
#define SIGMA 2.0f

// Function to create a Gaussian kernel
void createGaussianKernel(float *kernel, int size, float sigma)
{
    float sum = 0.0f;
    int halfSize = size / 2;

    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++)
        {
            kernel[(y + halfSize) * size + (x + halfSize)] =
                (1.0f / (2.0f * M_PI * sigma * sigma)) *
                expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            sum += kernel[(y + halfSize) * size + (x + halfSize)];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < size * size; i++)
    {
        kernel[i] /= sum;
    }
}

// Функция размытия для двумерного изображения
__global__ void blurKernel(unsigned char *inputImage,
                           unsigned char *outputImage, int width, int height,
                           float *kernel, int kernelSize)
{
    __shared__ unsigned char sharedMem[BLOCK_SIZE + 2 * FILTER_RADIUS + 1][BLOCK_SIZE + 2 * FILTER_RADIUS + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= width && y <= height)
    {
        sharedMem[threadIdx.y + FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = inputImage[y * width + x];
        // Заполнение левой границы
        if (threadIdx.x <= FILTER_RADIUS)
        {
            sharedMem[threadIdx.y + FILTER_RADIUS][threadIdx.x] = inputImage[y * width + x];
        }
        // Заполнение правой границы
        if (threadIdx.x >= BLOCK_SIZE - FILTER_RADIUS)
        {
            sharedMem[threadIdx.y + FILTER_RADIUS][threadIdx.x + 2 * FILTER_RADIUS] = inputImage[y * width + x];
        }
        // Заполнение верхней границы
        if (threadIdx.y <= FILTER_RADIUS)
        {
            sharedMem[threadIdx.y][threadIdx.x + FILTER_RADIUS] = inputImage[y * width + x];
        }
        // Заполнение нижней границы
        if (threadIdx.y >= BLOCK_SIZE - FILTER_RADIUS)
        {
            sharedMem[threadIdx.y + 2 * FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = inputImage[y * width + x];
        }
    }
    __syncthreads(); // Ждем, пока все потоки закончат загрузку

    // Проверка на границы
    if (x >= FILTER_RADIUS && x <= width - FILTER_RADIUS && y >= FILTER_RADIUS && y <= height - FILTER_RADIUS)
    {
        float sum = 0.0f;

        for (int dy = -FILTER_RADIUS - 1; dy <= FILTER_RADIUS + 1; dy++)
        {
            for (int dx = -FILTER_RADIUS - 1; dx <= FILTER_RADIUS + 1; dx++)
            {
                int nx = threadIdx.x + dx;
                int ny = threadIdx.y + dy;

                // Проверка на границы фильтра
                if (nx >= 0 && nx <= blockDim.x && ny >= 0 && ny <= blockDim.y)
                {
                    float kValue = kernel[(dy + FILTER_RADIUS) * kernelSize + (dx + FILTER_RADIUS)];

                    sum += (sharedMem[ny + FILTER_RADIUS][nx + FILTER_RADIUS] * kValue);
                }
            }
        }

        int index = y * width + x;
        outputImage[index] = static_cast<unsigned char>(sum); // Записываем в выходное значение
    }
}

void blurImage(unsigned char *h_inputImage,
               unsigned char *h_outputImage,
               int width, int height)
{
    unsigned char *d_inputImage, *d_outputImage;
    size_t imageSize = width * height;
    // Create Gaussian kernel
    float h_kernel[KERNEL_SIZE * KERNEL_SIZE];
    createGaussianKernel(h_kernel, KERNEL_SIZE, SIGMA);

    float *d_kernel;
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    blurKernel<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(unsigned char)>>>(d_inputImage, d_outputImage, width, height, d_kernel, KERNEL_SIZE);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

int main()
{

    Mat inputImage = imread("lena.jpg", IMREAD_GRAYSCALE);
    printf("%d", inputImage.cols);
    if (inputImage.empty())
    {
        cerr << "Ошибка загрузки изображения!" << endl;
        return -1;
    }

    const int width = inputImage.cols;  // 512
    const int height = inputImage.rows; // 512

    size_t imageSize = width * height;

    unsigned char *h_inputImage = inputImage.data; // Указатель на данные изображения
    unsigned char *h_outputImage = new unsigned char[imageSize];

    blurImage(h_inputImage, h_outputImage, width, height);

    // Сохраняем размазанное изображение
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("blurred_image.jpg", outputImage);

    delete[] h_outputImage;

    return 0;
}
