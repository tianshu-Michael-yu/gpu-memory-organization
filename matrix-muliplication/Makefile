NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAG = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

USE_CUDA = 0

ifeq ($(USE_CUDA), 0)
	MATRIX_MULTIPLICATION_OBJ = matrixMultiplication
else
	MATRIX_MULTIPLICATION_OBJ = matrixMultiplication_cu
endif

matrixMultiplication.exe: $(MATRIX_MULTIPLICATION_OBJ)

matrixMultiplication_cu: main.o matrixMultiplication_cu.o
	$(NVCC) $^ -o matrixMultiplication.exe

matrixMultiplication: main.o matrixMultiplication.o
	$(NVCC) $^ -o matrixMultiplication.exe

main.o: main.cpp matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

matrixMultiplication.o: matrixMultiplication.cpp matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

matrixMultiplication_cu.o: matrixMultiplication.cu matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

clean:
	rm -rf *.o *.exe