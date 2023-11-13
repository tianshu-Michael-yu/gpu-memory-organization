NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAG = -g -G -Xcompiler -Wall -gencode arch=compute_50,code=sm_50

all: main.exe

main.exe: kernel.cu
	$(NVCC) $(NVCC_FLAG) $^ -o $@