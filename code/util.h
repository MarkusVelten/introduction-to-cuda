#pragma once

#include <iostream>


#define checkCudaError(...) \
    checkCudaErrorImpl(__FILE__, __LINE__, __VA_ARGS__)

inline void checkCudaErrorImpl(const std::string &file, int line, cudaError_t code, bool checkGetLastError = false) {
    if (cudaSuccess != code) {
        std::cerr << "CUDA Error (" << file << " : " << line << ") --- " << cudaGetErrorString(code) << std::endl;
        exit(1);
    }

    if (checkGetLastError)
        checkCudaErrorImpl(file, line, cudaGetLastError(), false);
}


inline void initializeData(size_t* data, size_t numElements) {
    for (size_t i = 0; i < numElements; ++i)
        data[i] = i + 1337;
}


inline void verifyData(const size_t* data, size_t numElements, size_t numIterations) {
    for (int i = 0; i < numElements; ++i) {
        auto expected = (i + 1337) + numIterations;
        if (data[i] != expected) {
            std::cout << "Error at index " << i << ": got " << data[i] << ", expected " << expected << std::endl;
            return;
        }
    }

    std::cout << "Success" << std::endl;
}
