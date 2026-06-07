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

    //# TODO: for each work package (A/B/C/D), submit H2D, kernel, and D2H to the same stream.
    //#       Replace cudaMemcpy with cudaMemcpyAsync.

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
