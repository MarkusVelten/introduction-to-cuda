#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void fmaKernel(float* data, float scale, float add, int numFMA, int numElements) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
        float acc = data[i];
        for (int r = 0; r < numFMA; ++r)
            acc = scale * acc + add;
        data[i] = acc;
    }
}

int main() {
    const int numBlocks = 21;
    const int numThreadsPerBlock = 1024;
    const int numChunks = 16;
    const int totalElements = 128 * 4 * numBlocks * numThreadsPerBlock;  //# 4 matches the number of datasets in the running example
    const int chunkElements = totalElements / numChunks;
    const size_t chunkBytes = chunkElements * sizeof(float);

    float* data;
    checkCudaError(cudaMallocHost(&data, totalElements * sizeof(float)));

    float* d_data;
    checkCudaError(cudaMalloc(&d_data, totalElements * sizeof(float)));

    for (int i = 0; i < totalElements; ++i)
        data[i] = (float)i;

    //# heap-allocated stream array accommodates a variable numChunks
    cudaStream_t* streams = new cudaStream_t[numChunks];
    for (int i = 0; i < numChunks; ++i)
        checkCudaError(cudaStreamCreate(&streams[i]));

    //# run an empty kernel once to mitigate startup effects
    fmaKernel<<<1, 1>>>(0, 0, 0, 0, 0);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numChunks; ++i) {
        const int offset = i * chunkElements;
        checkCudaError(cudaMemcpyAsync(d_data + offset, data + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        fmaKernel<<<numBlocks, numThreadsPerBlock, 0, streams[i]>>>(d_data + offset, 0.12f, 1.2f, 1600, chunkElements);
        checkCudaError(cudaMemcpyAsync(data + offset, d_data + offset, chunkBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < numChunks; ++i)
        checkCudaError(cudaStreamSynchronize(streams[i]));

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Time elapsed: " << elapsed.count() * 1e3 << " ms\n";

    for (int i = 0; i < numChunks; ++i)
        checkCudaError(cudaStreamDestroy(streams[i]));
    delete[] streams;

    checkCudaError(cudaFreeHost(data));
    checkCudaError(cudaFree(d_data));
}
