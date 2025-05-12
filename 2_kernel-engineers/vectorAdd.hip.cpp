#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

// Define a simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    // Number of elements in the vectors
    int N = 1 << 20; // e.g., 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // Copy data from host to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the CUDA kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Copy the result back to host
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; i++) {
         if (fabs(h_C[i] - 3.0f) > 1e-5) {
             success = false;
             break;
         }
    }
    if (success)
        std::cout << "Vector addition is correct!" << std::endl;
    else
        std::cout << "Error in vector addition!" << std::endl;

    // Free device and host memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

