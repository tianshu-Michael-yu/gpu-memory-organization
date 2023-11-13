NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAG = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

USE_CUDA = 0

ifeq ($(USE_CUDA), 0)
	VECTOR_ADD_OBJ = vectorAdd
else
	VECTOR_ADD_OBJ = vectorAdd_cu
endif

vectorAdd.exe: $(VECTOR_ADD_OBJ)

vectorAdd_cu: main.o vectorAdd_cu.o
	$(NVCC) $^ -o vectorAdd.exe

vectorAdd: main.o vectorAdd.o
	$(NVCC) $^ -o vectorAdd.exe

main.o: main.cpp vectorAdd.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

vectorAdd.o: vectorAdd.cpp vectorAdd.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

vectorAdd_cu.o: vectorAdd.cu vectorAdd.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

clean:
	rm -rf *.o *.exe