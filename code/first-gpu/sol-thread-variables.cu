#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void kernel() {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < 10; i += stride)
        printf("Hello world from iteration %d\n", i);
}

int main(int argc, char *argv[]) {
    int numBlocks = 5;
    int numThreadsPerBlock = 2;

    kernel<<<numBlocks, numThreadsPerBlock>>>();
    cudaDeviceSynchronize();
}
