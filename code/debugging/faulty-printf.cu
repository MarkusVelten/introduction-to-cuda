#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void reduce(int* acc, int numElements) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < numElements; i += stride) {
        printf("[block %d, thread %d] Processing element %d; Accumulator: %d\n", blockIdx.x, threadIdx.x, i, acc[0]);
        acc[0] += 1;
        printf("[block %d, thread %d] Updated Accumulator: %d\n", blockIdx.x, threadIdx.x, acc[0]);
    }
}

int main(int argc, char *argv[]) {
    int numElements = 64;

    int acc = 0;

    int *d_acc;
    checkCudaError(cudaMalloc(&d_acc, sizeof(int)));

    //# reset accumulator
    checkCudaError(cudaMemset(d_acc, 0, sizeof(int)));

    //# run reduction
    reduce<<<2, 32>>>(d_acc, numElements);

    checkCudaError(cudaDeviceSynchronize(), true);

    //# copy data back to host
    checkCudaError(cudaMemcpy(&acc, d_acc, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Accumulator: " << acc << " (should be " << numElements << ")" << std::endl;

    checkCudaError(cudaFree(d_acc));
}
