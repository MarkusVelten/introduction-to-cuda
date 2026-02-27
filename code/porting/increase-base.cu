#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

int main(int argc, char *argv[]) {
    size_t numElements = 4 * 1024 * 1024;
    size_t numIterations = 8;

    size_t *data;
    data = new size_t[numElements];

    //# initialize data
    for (int i = 0; i < numElements; ++i) {
        data[i] = i + 1337;
    }

    //# main 'work'
    for (int it = 0; it < numIterations; ++it) {
        for (int i = 0; i < numElements; ++i) {
            data[i] += 1;
        }
    }

    //# verify results
    for (int i = 0; i < numElements; ++i) {
        auto expected = (i + 1337) + numIterations;
        if (data[i] != expected) {
            std::cout << "Error at index " << i << ": got " << data[i] << ", expected " << expected << std::endl;
            break;
        }

        if (numElements - 1 == i)
            std::cout << "Success" << std::endl;
    }

    delete[] data;
}
