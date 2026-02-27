#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void increase(size_t* data, size_t numElements) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = start; i < numElements; i += stride)
        data[i] += 1;
}

int main(int argc, char *argv[]) {
    size_t numElements = 4 * 1024 * 1024;
    size_t numIterations = 8;

    size_t *data;
    checkCudaError(cudaMallocHost(&data, numElements * sizeof(double)));
    size_t *d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(double)));

    initializeData(data, numElements);

    //# copy data to device
    checkCudaError(cudaMemcpy(d_data, data, numElements * sizeof(double), cudaMemcpyHostToDevice));

    //# main 'work'
    for (int it = 0; it < numIterations; ++it) {
        auto numBlocks = 108 * 32;
        auto numThreadsPerBlock = 256;
        increase<<<numBlocks, numThreadsPerBlock>>>(d_data, numElements);
    }
    checkCudaError(cudaDeviceSynchronize());

    //# copy data back to host
    checkCudaError(cudaMemcpy(data, d_data, numElements * sizeof(double), cudaMemcpyDeviceToHost));

    verifyData(data, numElements, numIterations);

    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaFreeHost(data));
}
