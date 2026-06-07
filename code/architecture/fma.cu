#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

constexpr size_t numFMA = 1024 * 1024;

__global__ void work(float* data, size_t numElements) {
    for (size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < numElements; i0 += blockDim.x * gridDim.x) {
        float acc = i0;

        for (auto r = 0; r < numFMA; ++r)
            acc = 0.12f * acc + 1.2f;

        //# dummy check to prevent compiler from eliminating loop
        if (0.f == acc)
            data[i0] = acc;
    }
}

int main(int argc, char *argv[]) {
    auto numBlocks = 84 * 0 + 1; //# start with one block, then vary the number of blocks to see how it affects performance
    auto numThreadsPerBlock = 64;
    auto numElements = numBlocks * numThreadsPerBlock;
    auto numIterations = 10;

    float *d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(float)));

    //# warm-up
    work<<<numBlocks, numThreadsPerBlock>>>(d_data, numElements);

    checkCudaError(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    //# main 'work'
    for (auto it = 0; it < numIterations; ++it) {
        work<<<numBlocks, numThreadsPerBlock>>>(d_data, numElements);
    }

    checkCudaError(cudaDeviceSynchronize(), true);
    auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsedSeconds = end - start;
    auto numFlopsPerElement = 2 * numFMA;
    std::cout << "Time elapsed:          " << elapsedSeconds.count() << " s\n";
    std::cout << "Estimated performance: " << 1e-12 * numFlopsPerElement * numElements * numIterations / elapsedSeconds.count() << " TFLOP/s\n";

    checkCudaError(cudaFree(d_data));
}
