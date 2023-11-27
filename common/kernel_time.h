#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { printf("\nCUDA Error: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

#define RECORD_KERNEL_TIME(x) {
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

    CUDA_CALL(cudaEventRecord(start));

    x;

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

    printf("Kernel time: %f ms\n", elapsed_time);
}