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
        h_data[i] = new float[numElements]();
        checkCudaError(cudaMalloc(&d_data[i], numElements * sizeof(float)));
    }

    //# TODO: create numArrays streams

    auto start = std::chrono::steady_clock::now();

    //# H2D transfers — default stream (unchanged)
    for (int i = 0; i < numArrays; ++i)
        checkCudaError(cudaMemcpy(d_data[i], h_data[i], numElements * sizeof(float), cudaMemcpyHostToDevice));

    //# TODO: launch each kernel in its own stream instead of the default stream
    for (int i = 0; i < numArrays; ++i)
        fmaKernel<<<numBlocks, numThreadsPerBlock>>>(d_data[i], numElements);

    //# TODO: synchronize all streams before the D2H transfers

    //# D2H transfers — default stream (unchanged)
    for (int i = 0; i < numArrays; ++i)
        checkCudaError(cudaMemcpy(h_data[i], d_data[i], numElements * sizeof(float), cudaMemcpyDeviceToHost));

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Time elapsed: " << elapsed.count() << " s\n";

    for (int i = 0; i < numArrays; ++i) {
        delete[] h_data[i];
        checkCudaError(cudaFree(d_data[i]));
    }

    //# TODO: destroy streams
}
