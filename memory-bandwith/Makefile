NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

bandwidthTest.exe: bandwidthTest.cu
	$(NVCC) $(NVCC_FLAGS) bandwidthTest.cu -o bandwidthTest.exe