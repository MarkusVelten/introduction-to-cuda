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
    cudaStream_t streams[numStreams];

    for (int s = 0; s < numStreams; ++s)
        checkCudaError(cudaStreamCreate(&streams[s]));

    for (int s = 0; s < numStreams; ++s)
        printStream<<<1, 1, 0, streams[s]>>>(s);

    for (int s = 0; s < numStreams; ++s)
        checkCudaError(cudaStreamSynchronize(streams[s]));

    for (int s = 0; s < numStreams; ++s)
        checkCudaError(cudaStreamDestroy(streams[s]));

    cudaDeviceSynchronize();
}
