#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void increase(size_t* data, size_t numElements) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < numElements; i += stride) {
        data[i] += 1;
    }
}

int main(int argc, char *argv[]) {
    size_t numElements = 4 * 1024 * 1024;
    size_t numIterations = 8;

    size_t *data;
    checkCudaError(cudaMallocManaged(&data, numElements * sizeof(size_t)));

    initializeData(data, numElements);

    //# TODO: prefetch data to the GPU

    //# main 'work'
    for (int it = 0; it < numIterations; ++it) {
        auto numBlocks = 84 * 32;
        auto numThreadsPerBlock = 256;
        increase<<<numBlocks, numThreadsPerBlock>>>(data, numElements);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    //# TODO: Bonus: prefetch data back to the CPU

    verifyData(data, numElements, numIterations);

    checkCudaError(cudaFree(data));
}
