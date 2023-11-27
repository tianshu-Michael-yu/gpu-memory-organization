#include "stdio.h"
#include "assert.h"

typedef unsigned short int u16;
typedef unsigned int u32;

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { printf("\nCUDA Error: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

#define KERNEL_LOOP 4096

__constant__ static u32 const_data_gpu[KERNEL_LOOP];
__device__ static u32 gmem_data_gpu[KERNEL_LOOP];
static u32 const_data_host[KERNEL_LOOP];

__global__ void const_test_gpu_gmem(u32 * const data, const u32 num_elements) {
    const u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_elements) {
        u32 d = gmem_data_gpu[0];

        for (int i=0; i<KERNEL_LOOP; i++) {
            d ^= gmem_data_gpu[i];
            d |= gmem_data_gpu[i];
            d &= gmem_data_gpu[i];
            d |= gmem_data_gpu[i];
        }

        data[tid] = d;
    }
}


__global__ void const_test_gpu_const(u32 * const data, const u32 num_elements) {
    const u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_elements) {
        u32 d = const_data_gpu[0];

        for (int i=0; i<KERNEL_LOOP; i++) {
            d ^= const_data_gpu[i];
            d |= const_data_gpu[i];
            d &= const_data_gpu[i];
            d |= const_data_gpu[i];
        }

        data[tid] = d;
    }
}



__host__ void cuda_error_check(const char * prefix, const char * postfix) {
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("%s%s%s\n", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
        cudaDeviceReset();
        exit(1);
    }
}


__host__ void generate_rand_data(u32 * host_data_ptr) {
    for (int i=0; i<KERNEL_LOOP; i++) {
        host_data_ptr[i] = rand();
    }
}


__host__ void gpu_kernel(void) {
    const u32 num_elements = (128*1024);
    const u32 num_threads = 256;
    const u32 num_blocks = (num_elements + num_threads - 1) / num_threads;
    const u32 num_bytes = num_elements * sizeof(u32);
    int max_device_num;
    const int max_runs = 6;

    CUDA_CALL(cudaGetDeviceCount(&max_device_num));

    for (int device_num = 0; device_num < max_device_num; device_num++) {
        CUDA_CALL(cudaSetDevice(device_num));

        u32 * data_gpu;
        cudaEvent_t kernel_start1, kernel_stop1;
        cudaEvent_t kernel_start2, kernel_stop2;
        float delta_time1 = 0.0F, delta_time2 = 0.0F;
        struct cudaDeviceProp device_prop;
        char device_prefix[261];

        CUDA_CALL(cudaMalloc(&data_gpu, num_bytes));
        CUDA_CALL(cudaEventCreate(&kernel_start1));
        CUDA_CALL(cudaEventCreate(&kernel_start2));
        CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop1, cudaEventBlockingSync));
        CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop2, cudaEventBlockingSync));

        CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_num));
        sprintf(device_prefix, "ID: %d (%s): ", device_num, device_prop.name);

        for (int num_test=0; num_test < max_runs; num_test++) {
            // Generate random data
            generate_rand_data(const_data_host);

            // Copy host memory to constant memory section in GPU
            CUDA_CALL(cudaMemcpyToSymbol(const_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32)));
            // Warm up run
            const_test_gpu_gmem<<<num_blocks, num_threads>>>(data_gpu, num_elements);
            cuda_error_check("Error ", " returned from gmem startup kernel");

            // Do the gmem kernel
            CUDA_CALL(cudaEventRecord(kernel_start1, 0));

            const_test_gpu_gmem<<<num_blocks, num_threads>>>(data_gpu, num_elements);

            cuda_error_check("Error ", " returned from gmem kernel");
            CUDA_CALL(cudaEventRecord(kernel_stop1, 0));
            CUDA_CALL(cudaEventSynchronize(kernel_stop1));
            CUDA_CALL(cudaEventElapsedTime(&delta_time1, kernel_start1, kernel_stop1));

            // Copy host memory to global memory section in GPU
            CUDA_CALL(cudaMemcpyToSymbol(gmem_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32)));

            // Warm up run
            const_test_gpu_const<<<num_blocks, num_threads>>>(data_gpu, num_elements);
            cuda_error_check("Error ", " returned from const startup kernel");

            // Do the const kernel
            CUDA_CALL(cudaEventRecord(kernel_start2, 0));

            const_test_gpu_const<<<num_blocks, num_threads>>>(data_gpu, num_elements);
            cuda_error_check("Error ", " returned from const kernel");

            CUDA_CALL(cudaEventRecord(kernel_stop2, 0));
            CUDA_CALL(cudaEventSynchronize(kernel_stop2));
            CUDA_CALL(cudaEventElapsedTime(&delta_time2, kernel_start2, kernel_stop2));

            if (delta_time1 > delta_time2) {
                printf("\n%sConstant version is faster by: %.2fms (G=%.2fms, C=%.2fms)", device_prefix, delta_time1 - delta_time2, delta_time1, delta_time2);
            } else {
                printf("\n%sGlobal version is faster by: %.2fms (G=%.2fms, C=%.2fms)", device_prefix, delta_time2 - delta_time1, delta_time1, delta_time2);
            }

            CUDA_CALL(cudaEventDestroy(kernel_start1));
            CUDA_CALL(cudaEventDestroy(kernel_start2));
            CUDA_CALL(cudaEventDestroy(kernel_stop1));
            CUDA_CALL(cudaEventDestroy(kernel_stop2));

            CUDA_CALL(cudaDeviceReset());
            printf("\n");
        }

    }
}

int main(void) {
    gpu_kernel();
    return EXIT_SUCCESS;
}
