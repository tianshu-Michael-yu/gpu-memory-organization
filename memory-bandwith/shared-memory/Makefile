NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

smem.exe: smem.cu
	$(NVCC) $(NVCC_FLAGS) smem.cu -o smem.exe