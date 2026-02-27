#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void kernel() {
    printf("Hello world from thread %d\n", threadIdx.x);
}

int main(int argc, char *argv[]) {
    int numThreads = 8;
    kernel<<<1, numThreads>>>();
    checkCudaError(cudaDeviceSynchronize(), true);
}
