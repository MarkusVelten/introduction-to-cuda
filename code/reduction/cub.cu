#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

#include <cub/cub.cuh>

__global__ void reduce(int* acc, int numElements) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int thread_acc = 0;

    for (int i = start; i < numElements; i += stride)
        thread_acc += 1;

    //# use cub for a block-wide reduction
    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    int block_acc = BlockReduce(tempStorage).Sum(thread_acc);

    //# only the first thread in the current block receives the block-wide result
    //# and adds it to the global accumulator
    if (0 == threadIdx.x) {
        atomicAdd(&acc[0], block_acc);
    }
}

int main(int argc, char *argv[]) {
    int numElements = 4 * 1024 * 1024;

    int acc = 0;

    int *d_acc;
    cudaMalloc(&d_acc, sizeof(int));

    //# warm-up
    reduce<<<84 * 32, 256>>>(d_acc, numElements);

    //# reset accumulator
    cudaMemset(d_acc, 0, sizeof(int));

    auto start = std::chrono::steady_clock::now();

    //# run reduction
    reduce<<<84 * 32, 256>>>(d_acc, numElements);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "Time elapsed:          " << elapsedSeconds.count() << " s\n";
    std::cout << "Estimated performance: " << 1e-9 * numElements / elapsedSeconds.count() << " GFLOP/s\n";

    //# copy data back to host
    cudaMemcpy(&acc, d_acc, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Accumulator: " << acc << " (should be " << numElements << ")" << std::endl;

    cudaFree(d_acc);
}
