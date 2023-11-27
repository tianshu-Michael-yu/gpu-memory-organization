NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

registers.exe: registers.cu
	$(NVCC) $(NVCC_FLAGS) registers.cu -o registers.exe