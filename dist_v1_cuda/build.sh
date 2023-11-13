/usr/local/cuda/bin/nvcc -ccbin gcc -gencode arch=compute_50,code=sm_50 -o vectorAdd.o -c vectorAdd.cu
/usr/local/cuda/bin/nvcc -ccbin gcc -o vectorAdd vectorAdd.o 