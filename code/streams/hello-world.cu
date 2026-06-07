#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void printStream(int streamId) {
    printf("Hello world from stream %d\n", streamId);
}

int main() {
    const int numStreams = 5;

    for (int s = 0; s < numStreams; ++s) {
        cudaStream_t stream;

        checkCudaError(cudaStreamCreate(&stream));
        printStream<<<1, 1, 0, stream>>>(s);
        checkCudaError(cudaStreamDestroy(stream));
    }

    cudaDeviceSynchronize();
}
