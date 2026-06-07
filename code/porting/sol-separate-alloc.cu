#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void increase(size_t* data, size_t numElements) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < numElements; i += stride)
        data[i] += 1;
}

int main(int argc, char *argv[]) {
    size_t numElements = 4 * 1024 * 1024;
    size_t numIterations = 8;

    size_t *data;
    checkCudaError(cudaMallocHost(&data, numElements * sizeof(size_t)));
    size_t *d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(size_t)));

    initializeData(data, numElements);

    //# copy data to device
    checkCudaError(cudaMemcpy(d_data, data, numElements * sizeof(size_t), cudaMemcpyHostToDevice));

    //# main 'work'
    for (int it = 0; it < numIterations; ++it) {
        auto numBlocks = 84 * 32;
        auto numThreadsPerBlock = 256;
        increase<<<numBlocks, numThreadsPerBlock>>>(d_data, numElements);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    //# copy data back to host
    checkCudaError(cudaMemcpy(data, d_data, numElements * sizeof(size_t), cudaMemcpyDeviceToHost));

    verifyData(data, numElements, numIterations);

    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaFreeHost(data));
}
