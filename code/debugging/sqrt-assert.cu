#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

#include <cassert>

__global__ void sqrtKernel(float* data, int numElements) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
        assert(data[i] >= 0.);      //# triggers for negative values

        data[i] = sqrt(data[i]);
    }
}

int main() {
    const int numElements = 5;
    float h_data[] = {4., 9., -1., 16., 25.};       //# -1.0 triggers the assert

    float* d_data;
    checkCudaError(cudaMalloc(&d_data, numElements * sizeof(float)));
    checkCudaError(cudaMemcpy(d_data, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice));

    sqrtKernel<<<1, 32>>>(d_data, numElements);
    checkCudaError(cudaDeviceSynchronize(), true);  //# reports assertion failure

    checkCudaError(cudaFree(d_data));
}
