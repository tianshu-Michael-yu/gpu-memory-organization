#include "matrixMultiplication.h"

void matrixMultiplication(int *matrixA, int *matrixB, int *matrixC, size_t size) {
    #define IND(row, col) ((row) * (size) + (col))
    for (size_t k=0; k < size; ++k) {
        for (size_t i=0; i < size; ++i) {
            for (size_t j=0; j < size; ++j) {
                matrixC[IND(i, j)] += matrixA[IND(i, k)] * matrixB[IND(k, j)];
            }
        }
    }
}
