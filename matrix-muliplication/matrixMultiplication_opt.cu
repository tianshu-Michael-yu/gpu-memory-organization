#include "matrixMultiplication.h"
#include <stdio.h>

__global__ void matrixMulKernel(int *d_A, int *d_B, int *d_C, size_t size) {
    int i= blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && k < size) {
        for (int j=0; j<size; j++) {
            atomicAdd(&d_C[i*size + j], d_A[i*size + k] * d_B[k*size + j]);
        }
    }
}


void matrixMultiplication(int *matrixA, int *matrixB, int *matrixC, size_t matrixSize) {
    int *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaError_t err1 = cudaMalloc((void **) &d_A, matrixSize * matrixSize * sizeof(*d_A));
    cudaError_t err2 = cudaMalloc((void **) &d_B, matrixSize * matrixSize * sizeof(*d_B));
    cudaError_t err3 = cudaMalloc((void **) &d_C, matrixSize * matrixSize * sizeof(*d_C));

    // error checking
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("Error allocating memory on the device\n");
        exit(EXIT_FAILURE);
    }

    // Copy the matrices to the device
    cudaError_t err4 = cudaMemcpy(d_A, matrixA, matrixSize * matrixSize * sizeof(*d_A), cudaMemcpyHostToDevice);
    cudaError_t err5 = cudaMemcpy(d_B, matrixB, matrixSize * matrixSize * sizeof(*d_B), cudaMemcpyHostToDevice);



    // error checking
    if (err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("Error copying memory to the device\n");
        exit(EXIT_FAILURE);
    }
    
    // Launch the kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((matrixSize + dimBlock.x - 1) / dimBlock.x, (matrixSize + dimBlock.y - 1) / dimBlock.y);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, matrixSize);

    // Copy the result back to the host
    cudaError_t err6 = cudaMemcpy(matrixC, d_C, matrixSize * matrixSize * sizeof(*d_C), cudaMemcpyDeviceToHost);

    // error checking
    if (err6 != cudaSuccess) {
        printf("Error copying memory back to the host\n");
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaError_t err7 = cudaFree(d_A);
    cudaError_t err8 = cudaFree(d_B);
    cudaError_t err9 = cudaFree(d_C);

    // error checking
    if (err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("Error freeing memory on the device\n");
        exit(EXIT_FAILURE);
    }
}