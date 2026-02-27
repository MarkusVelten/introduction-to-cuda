#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

constexpr size_t numFMA = 1024 * 1024;

__global__ void fma(float* data, size_t numElements) {
    for (size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < numElements; i0 += blockDim.x * gridDim.x) {
        float acc = i0;

        for (auto r = 0; r < numFMA; ++r)
            acc = 0.12f * acc + 1.2f;

        //# dummy check to prevent compiler from eliminating loop
        if (0.0 == acc)
            data[i0] = acc;
    }
}

int main(int argc, char *argv[]) {
    auto numBlocks = 84 * 3 + 1;
    auto numThreadsPerBlock = 256;
    auto numElements = numBlocks * numThreadsPerBlock;
    auto numIterations = 10;

    float *d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(float)));

    //# warm-up
    fma<<<numBlocks, numThreadsPerBlock>>>(d_data, numElements);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    //# main 'work'
    for (auto it = 0; it < numIterations; ++it) {
        fma<<<numBlocks, numThreadsPerBlock>>>(d_data, numElements);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsedSeconds = end - start;
    auto numFlopsPerElement = 2 * numFMA;
    std::cout << "Time elapsed:          " << elapsedSeconds.count() << " s\n";
    std::cout << "Estimated performance: " << 1e-12 * numFlopsPerElement * numElements * numIterations / elapsedSeconds.count() << " TFLOP/s\n";

    cudaFree(d_data);
}
