NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAG = -O3 -gencode arch=compute_50,code=sm_50 -ccbin gcc

all: matrixMultiplication.exe matrixMultiplication_cu.exe matrixMultiplication_opt.exe matrixMultiplication_serial_opt.exe

matrixMultiplication_serial_opt.exe: main.o matrixMultiplication_serial_opt.o
	$(NVCC) $^ -o matrixMultiplication_serial_opt.exe

matrixMultiplication_opt.exe: main.o matrixMultiplication_opt.o
	$(NVCC) $^ -o matrixMultiplication_opt.exe

matrixMultiplication_cu.exe: main.o matrixMultiplication_cu.o
	$(NVCC) $^ -o matrixMultiplication_cu.exe

matrixMultiplication.exe: main.o matrixMultiplication.o
	$(NVCC) $^ -o matrixMultiplication.exe

main.o: main.cpp matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

matrixMultiplication.o: matrixMultiplication.cpp matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

matrixMultiplication_cu.o: matrixMultiplication.cu matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

matrixMultiplication_opt.o: matrixMultiplication_opt.cu matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

matrixMultiplication_serial_opt.o: matrixMultiplication_serial_opt.cpp matrixMultiplication.h
	$(NVCC) $(NVCC_FLAG) -c $< -o $@

clean:
	rm -rf *.o *.exe