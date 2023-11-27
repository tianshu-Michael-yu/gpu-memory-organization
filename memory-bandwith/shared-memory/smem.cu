#include "stdio.h"
#include "assert.h"

typedef unsigned short int u16;
typedef unsigned int u32;

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { printf("\nCUDA Error: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

#define KERNEL_LOOP 4096

__shared__ u32 smem_data_gpu[KERNEL_LOOP];

__global__ void const_test_gpu_smem(u32 * const data, const u32 num_elements) {
    const u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    smem_data_gpu[0] = 0;
    if (tid < num_elements) {
        u32 d = 0;

        for (int i=0; i<KERNEL_LOOP; i++) {
            smem_data_gpu[(i+1)%KERNEL_LOOP] = tid + i;
            d ^= smem_data_gpu[i];
            d |= smem_data_gpu[i];
            d &= smem_data_gpu[i];
        }

        data[tid] = d;
    }
}

void cuda_error_check() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        assert(0);
    }
}

__host__ void generate_rand_data(u32 * host_data_ptr) {
    for (int i=0; i<KERNEL_LOOP; i++) {
        host_data_ptr[i] = rand();
    }
}


int main() {
    const u32 num_elements = 1 << 20;
    const u32 threads_per_block = 256;
    const u32 num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;


    u32 * data_gpu;
    CUDA_CALL(cudaMalloc(&data_gpu, sizeof(*data_gpu) * num_elements));

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

    CUDA_CALL(cudaEventRecord(start));
    const_test_gpu_smem<<<num_blocks, threads_per_block>>>(data_gpu, num_elements);
    cuda_error_check();

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

    printf("shared memory time: %f ms\n", elapsed_time);

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    CUDA_CALL(cudaFree(data_gpu));

    return EXIT_SUCCESS;
}