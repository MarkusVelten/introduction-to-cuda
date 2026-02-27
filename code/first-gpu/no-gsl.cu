#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 10)
        printf("Hello world from iteration %d\n", tid);
}

int main(int argc, char *argv[]) {
    int numThreadsPerBlock = 5;
    int numBlocks = cuda::ceil_div(10, numThreadsPerBlock);

    kernel<<<numBlocks, numThreadsPerBlock>>>();
    cudaDeviceSynchronize();
}
