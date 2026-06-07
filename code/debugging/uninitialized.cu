#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void addOne(int* data, int numElements) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
        data[idx] += 1;
}

int main() {
    const int numElements = 2;

    int* d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(int)));

    addOne<<<1, 32>>>(d_data, numElements);
    checkCudaError(cudaDeviceSynchronize(), true);

    std::cout << "Kernel completed" << std::endl;

    checkCudaError(cudaFree(d_data));
}
