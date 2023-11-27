NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

constantMemory.exe: constant-memory.cu
	$(NVCC) $(NVCC_FLAGS) constant-memory.cu -o constantMemory.exe