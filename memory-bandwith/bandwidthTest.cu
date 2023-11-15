#include <stdio.h>
#define TPB 1024

__global__ void bandwith(int *array) {
    int i = threadIdx.x;
    array[i] = i;
    array[i] = i+1;
    array[i] = i+2;
    array[i] = i+3;
    array[i] = i+4;
    array[i] = i+5;
}

void bandwithTest() {
    int *host_array = (int*)malloc(TPB * sizeof(*host_array));

    int *array;
    cudaEvent_t startKernel, stopKernel;
    cudaEventCreate(&startKernel); 
    cudaEventCreate(&stopKernel);

    cudaMalloc(&array, TPB * sizeof(*array));

    cudaEventRecord(startKernel);
    bandwith<<<1, TPB>>>(array); // test warp memory write bandwith
    cudaEventRecord(stopKernel);

    // copy back the results
    cudaMemcpy(host_array, array, TPB * sizeof(int), cudaMemcpyDeviceToHost);

    // check the results
    for (size_t i = 0; i < TPB; ++i) {
        if (host_array[i] != i+5) {
            printf("Error at index %d: Expected %d, got %d\n", i, i+5, host_array[i]);
            exit(EXIT_FAILURE);
        }
    }

    cudaEventSynchronize(stopKernel);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);
    printf("Kernel time: %f ms\n", kernelTime);
    float memoryBandwidth = (TPB * sizeof(int) * 6) / (kernelTime * 1e6);
    printf("Memory bandwidth: %f GB/s\n", memoryBandwidth);
    cudaFree(array);
    free(host_array);
}

int main() {
    bandwithTest();
    return EXIT_SUCCESS;
}