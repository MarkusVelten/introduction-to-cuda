#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    std::cout << "\nDevice " << device << ": " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "  Global Memory: " << std::fixed << std::setprecision(2)
              << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Warp Size: " << prop.warpSize << "\n";
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max Threads Dim: [" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  Max Grid Size: [" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";

    return 0;
}
