NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

cmem.exe: cmem.cu
	$(NVCC) $(NVCC_FLAGS) cmem.cu -o cmem.exe