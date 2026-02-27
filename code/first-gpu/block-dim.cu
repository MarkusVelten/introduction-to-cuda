#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void kernel() {
    int start = threadIdx.x;
    int numThreads = blockDim.x;

    for (int i = start; i < 10; i += numThreads)
        printf("Hello world from iteration %d\n", i);
}

int main(int argc, char *argv[]) {
    int numThreads = 2;
    kernel<<<1, numThreads>>>();
    cudaDeviceSynchronize();
}
