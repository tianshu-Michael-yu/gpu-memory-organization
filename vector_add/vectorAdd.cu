#include "vectorAdd.h"
#include <time.h>
#include <stdio.h>
#define TPB 1024

__global__ void vectorAddKernel(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vectorAdd(int *a, int *b, int *c, int n) {
    int *dev_a, *dev_b, *dev_c;

    // allocate memory on device
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));
    cudaMalloc((void**)&dev_c, n * sizeof(int));

    // copy vectors from host memory to device memory
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    clock_t kernel_start = clock();
    // launch kernel
    vectorAddKernel<<<ceil(n / (float)TPB), TPB>>>(dev_a, dev_b, dev_c, n);
    cudaDeviceSynchronize();
    clock_t kernel_end = clock();
    printf("Kernel time: %f\n", (double)(kernel_end - kernel_start) / CLOCKS_PER_SEC);

    // copy result from device memory to host memory
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}