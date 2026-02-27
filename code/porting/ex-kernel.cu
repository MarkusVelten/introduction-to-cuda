#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

void increase(size_t* data, size_t numElements) {
    for (int i = 0; i < numElements; ++i) {
        data[i] += 1;
    }
}

int main(int argc, char *argv[]) {
    size_t numElements = 4 * 1024 * 1024;
    size_t numIterations = 8;

    size_t *data;
    cudaMallocManaged(&data, numElements * sizeof(double));

    initializeData(data, numElements);

    //# main 'work'
    for (int it = 0; it < numIterations; ++it) {
        increase(data, numElements);
    }

    verifyData(data, numElements, numIterations);

    cudaFree(data);
}
