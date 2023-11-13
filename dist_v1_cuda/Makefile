NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -ccbin gcc -gencode arch=compute_50,code=sm_50 

main.exe: kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f *.o *.exe