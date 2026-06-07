constexpr int numFMA = 1024 * 1024;

__global__ void fmaKernel(float* data, int numElements) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x) {
        float acc = i;
        for (int r = 0; r < numFMA; ++r)
            acc = 0.12f * acc + 1.2f;
        if (acc == 0.f)
            data[i] = acc;
    }
}

int main() {
    const int numBlocks = 21;
    const int numThreadsPerBlock = 256;
    const int numElements = numBlocks * numThreadsPerBlock;
    const int numArrays = 4;

    float* h_data[numArrays];
    float* d_data[numArrays];
    for (int i = 0; i < numArrays; ++i) {
        checkCudaError(cudaMallocHost(&h_data[i], numElements * sizeof(float)));
        checkCudaError(cudaMalloc(&d_data[i], numElements * sizeof(float)));
    }

    cudaStream_t streams[numArrays];
    for (int i = 0; i < numArrays; ++i)
        checkCudaError(cudaStreamCreate(&streams[i]));

    auto start = std::chrono::steady_clock::now();

    //# H2D, kernel, and D2H for each work package — all in the same stream
    for (int i = 0; i < numArrays; ++i) {
        const size_t bytes = numElements * sizeof(float);
        checkCudaError(cudaMemcpyAsync(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice, streams[i]));
        fmaKernel<<<numBlocks, numThreadsPerBlock, 0, streams[i]>>>(d_data[i], numElements);
        checkCudaError(cudaMemcpyAsync(h_data[i], d_data[i], bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < numArrays; ++i)
        checkCudaError(cudaStreamSynchronize(streams[i]));

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Time elapsed: " << elapsed.count() << " s\n";

    for (int i = 0; i < numArrays; ++i) {
        checkCudaError(cudaFreeHost(h_data[i]));
        checkCudaError(cudaFree(d_data[i]));
    }

    for (int i = 0; i < numArrays; ++i)
        checkCudaError(cudaStreamDestroy(streams[i]));
}
