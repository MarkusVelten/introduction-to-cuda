#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void fmaKernel(float* data, float scale, float add, int numFMA, int numElements) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
        float acc = data[i];
        for (int r = 0; r < numFMA; ++r)
            acc = scale * acc + add;
        data[i] = acc;
    }
}

int main() {
    const int numBlocks = 21;
    const int numThreadsPerBlock = 1024;
    const int numElements = 128 * numBlocks * numThreadsPerBlock;

    float *dataA, *dataB, *dataC, *dataD;
    float *d_dataA, *d_dataB, *d_dataC, *d_dataD;

    for (auto p : {&dataA, &dataB, &dataC, &dataD})
        checkCudaError(cudaMallocHost(p, numElements * sizeof(float)));

    for (auto p : {&d_dataA, &d_dataB, &d_dataC, &d_dataD})
        checkCudaError(cudaMalloc(p, numElements * sizeof(float)));

    for (int i = 0; i < numElements; ++i)
        dataA[i] = dataB[i] = dataC[i] = dataD[i] = (float)i;

    cudaStream_t streamA, streamB, streamC, streamD;
    for (auto p : {&streamA, &streamB, &streamC, &streamD})
        checkCudaError(cudaStreamCreate(p));

    //# run an empty kernel once to mitigate startup effects
    fmaKernel<<<1, 1>>>(0, 0, 0, 0, 0);

    auto start = std::chrono::steady_clock::now();

    //# H2D, kernel, D2H for each work package - each submitted to its own stream
    checkCudaError(cudaMemcpyAsync(d_dataA, dataA, numElements * sizeof(float), cudaMemcpyHostToDevice, streamA));
    fmaKernel<<<numBlocks, numThreadsPerBlock, 0, streamA>>>(d_dataA, 0.11f, 1.1f, 1100, numElements);
    checkCudaError(cudaMemcpyAsync(dataA, d_dataA, numElements * sizeof(float), cudaMemcpyDeviceToHost, streamA));

    checkCudaError(cudaMemcpyAsync(d_dataB, dataB, numElements * sizeof(float), cudaMemcpyHostToDevice, streamB));
    fmaKernel<<<numBlocks, numThreadsPerBlock, 0, streamB>>>(d_dataB, 0.12f, 1.2f, 1200, numElements);
    checkCudaError(cudaMemcpyAsync(dataB, d_dataB, numElements * sizeof(float), cudaMemcpyDeviceToHost, streamB));

    checkCudaError(cudaMemcpyAsync(d_dataC, dataC, numElements * sizeof(float), cudaMemcpyHostToDevice, streamC));
    fmaKernel<<<numBlocks, numThreadsPerBlock, 0, streamC>>>(d_dataC, 0.13f, 1.3f, 1300, numElements);
    checkCudaError(cudaMemcpyAsync(dataC, d_dataC, numElements * sizeof(float), cudaMemcpyDeviceToHost, streamC));

    checkCudaError(cudaMemcpyAsync(d_dataD, dataD, numElements * sizeof(float), cudaMemcpyHostToDevice, streamD));
    fmaKernel<<<numBlocks, numThreadsPerBlock, 0, streamD>>>(d_dataD, 0.14f, 1.4f, 1400, numElements);
    checkCudaError(cudaMemcpyAsync(dataD, d_dataD, numElements * sizeof(float), cudaMemcpyDeviceToHost, streamD));

    for (auto p : {streamA, streamB, streamC, streamD})
        checkCudaError(cudaStreamSynchronize(p));

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Time elapsed: " << elapsed.count() * 1e3 << " ms\n";

    for (auto p : {dataA, dataB, dataC, dataD})
        checkCudaError(cudaFreeHost(p));

    for (auto p : {d_dataA, d_dataB, d_dataC, d_dataD})
        checkCudaError(cudaFree(p));

    for (auto p : {streamA, streamB, streamC, streamD})
        checkCudaError(cudaStreamDestroy(p));
}
