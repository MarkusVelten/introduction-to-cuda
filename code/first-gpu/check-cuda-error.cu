#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

__global__ void kernel() {
    for (int i = 0; i < 10; ++i)
        printf("Hello world from iteration %d\n", i);
}

int main(int argc, char *argv[]) {
    int numThreads = 0;
    kernel<<<1, numThreads>>>();
    checkCudaError(cudaDeviceSynchronize(), true);
}
