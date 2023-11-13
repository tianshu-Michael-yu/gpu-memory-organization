NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
main.exe: main.cpp
	$(NVCC) $(NVCC_FLAGS) -o $@ $^