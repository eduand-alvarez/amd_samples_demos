#include "hip/hip_runtime.h"
// relu_kernel.cu
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

__global__ void relu(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = fmaxf(0.0f, input[idx]);
}

void print_activations(float* data, int N) {
    for (int i = 0; i < N; ++i) {
        int bar_len = (int)(data[i] * 5);  // Scale for printing
        printf("[%2d] %6.2f | ", i, data[i]);
        for (int j = 0; j < bar_len; ++j) printf("#");
        printf("\n");
    }
}

int main() {
    const int N = 16;
    float h_input[N], h_output[N];

    // Simulate pre-activation values (e.g., layer output before ReLU)
    printf("🔢 Random 'pre-activation' values:\n");
    for (int i = 0; i < N; ++i) {
        h_input[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f; // range [-2, 2]
        printf("%6.2f ", h_input[i]);
    }
    printf("\n\n");

    float *d_input, *d_output;
    hipMalloc((void**)&d_input, N * sizeof(float));
    hipMalloc((void**)&d_output, N * sizeof(float));

    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);

    relu<<<1, N>>>(d_input, d_output, N);
    hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);

    printf("⚡ ReLU-activated outputs:\n");
    print_activations(h_output, N);

    return 0;
}

