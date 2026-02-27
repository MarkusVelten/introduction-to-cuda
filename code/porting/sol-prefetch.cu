#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void increase(size_t* data, size_t numElements) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = start; i < numElements; i += stride) {
        data[i] += 1;
    }
}

int main(int argc, char *argv[]) {
    size_t numElements = 4 * 1024 * 1024;
    size_t numIterations = 8;

    size_t *data;
    checkCudaError(cudaMallocManaged(&data, numElements * sizeof(double)));

    initializeData(data, numElements);

    int device;
    cudaGetDevice(&device);

    //# old version: checkCudaError(cudaMemPrefetchAsync(data, numElements * sizeof(double), device));
    checkCudaError(cudaMemPrefetchAsync(data, numElements * sizeof(double), { .type = cudaMemLocationTypeDevice, .id = device }, 0));

    //# main 'work'
    for (int it = 0; it < numIterations; ++it) {
        auto numBlocks = 84 * 32;
        auto numThreadsPerBlock = 256;
        increase<<<numBlocks, numThreadsPerBlock>>>(data, numElements);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    //# old version: checkCudaError(cudaMemPrefetchAsync(data, numElements * sizeof(double), cudaCpuDeviceId));
    checkCudaError(cudaMemPrefetchAsync(data, numElements * sizeof(double), { .type = cudaMemLocationTypeHost }, 0));

    verifyData(data, numElements, numIterations);

    checkCudaError(cudaFree(data));
}
