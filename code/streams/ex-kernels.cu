#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void fmaKernel(float* data, float scale, float add, int numFMA, int numElements) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
        float acc = data[i];
        for (int r = 0; r < numFMA; ++r)
            acc = scale * acc + add;
        data[i] = acc;
    }
}

int main() {
    const int numBlocks = 21;
    const int numThreadsPerBlock = 1024;
    const int numElements = 128 * numBlocks * numThreadsPerBlock;

    float *dataA, *dataB, *dataC, *dataD;
    float *d_dataA, *d_dataB, *d_dataC, *d_dataD;

    for (auto p : {&dataA, &dataB, &dataC, &dataD})
        checkCudaError(cudaMallocHost(p, numElements * sizeof(float)));

    for (auto p : {&d_dataA, &d_dataB, &d_dataC, &d_dataD})
        checkCudaError(cudaMalloc(p, numElements * sizeof(float)));

    for (int i = 0; i < numElements; ++i)
        dataA[i] = dataB[i] = dataC[i] = dataD[i] = (float)i;

    //# TODO: create four streams (streamA, streamB, streamC, streamD)

    //# run an empty kernel once to mitigate startup effects - nullptr will never be accessed
    fmaKernel<<<1, 1>>>(nullptr, 0, 0, 0, 0);

    auto start = std::chrono::steady_clock::now();

    checkCudaError(cudaMemcpy(d_dataA, dataA, numElements * sizeof(float), cudaMemcpyHostToDevice));
    //# TODO: launch fmaKernel for dataA in streamA instead of the default stream
    fmaKernel<<<numBlocks, numThreadsPerBlock>>>(d_dataA, 0.11f, 1.1f, 1100, numElements);
    checkCudaError(cudaMemcpy(dataA, d_dataA, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaError(cudaMemcpy(d_dataB, dataB, numElements * sizeof(float), cudaMemcpyHostToDevice));
    //# TODO: launch fmaKernel for dataB in streamB instead of the default stream
    fmaKernel<<<numBlocks, numThreadsPerBlock>>>(d_dataB, 0.12f, 1.2f, 1200, numElements);
    checkCudaError(cudaMemcpy(dataB, d_dataB, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaError(cudaMemcpy(d_dataC, dataC, numElements * sizeof(float), cudaMemcpyHostToDevice));
    //# TODO: launch fmaKernel for dataC in streamC instead of the default stream
    fmaKernel<<<numBlocks, numThreadsPerBlock>>>(d_dataC, 0.13f, 1.3f, 1300, numElements);
    checkCudaError(cudaMemcpy(dataC, d_dataC, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaError(cudaMemcpy(d_dataD, dataD, numElements * sizeof(float), cudaMemcpyHostToDevice));
    //# TODO: launch fmaKernel for dataD in streamD instead of the default stream
    fmaKernel<<<numBlocks, numThreadsPerBlock>>>(d_dataD, 0.14f, 1.4f, 1400, numElements);
    checkCudaError(cudaMemcpy(dataD, d_dataD, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Time elapsed: " << elapsed.count() * 1e3 << " ms\n";

    for (auto p : {dataA, dataB, dataC, dataD})
        checkCudaError(cudaFreeHost(p));

    for (auto p : {d_dataA, d_dataB, d_dataC, d_dataD})
        checkCudaError(cudaFree(p));

    //# TODO: destroy streams
}
