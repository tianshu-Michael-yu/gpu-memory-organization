NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

gmem.exe: gmem.cu
	$(NVCC) $(NVCC_FLAGS) gmem.cu -o gmem.exe