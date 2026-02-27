#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void kernel() {
    int start = //# TODO: calculate global thread index
    int stride = //# TODO: calculate total number of threads

    for (int i = start; i < 10; i += stride)
        printf("Hello world from iteration %d\n", i);
}

int main(int argc, char *argv[]) {
    int numBlocks = //# TODO: choose number of blocks
    int numThreadsPerBlock = //# TODO: choose number of threads per block

    kernel<<<numBlocks, numThreadsPerBlock>>>();
    cudaDeviceSynchronize();
}
