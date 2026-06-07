#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void fillArray(int* data, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //# BUG: if (idx < numElements) check is missing
    data[idx] = idx;
}

int main() {
    const int numElements = 30;

    int* d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(int)));

    fillArray<<<1, 32>>>(d_data, numElements);
    checkCudaError(cudaDeviceSynchronize(), true);

    std::cout << "Kernel completed" << std::endl;

    checkCudaError(cudaFree(d_data));
}
