#include "matrixMultiplication.h"

void matrixMultiplication(int *matrixA, int *matrixB, int *matrixC, size_t size) {
    #define IND(row, col) ((row) * (size) + (col))
    const size_t subBlockSize = 32;
    for (size_t i = 0; i < size; i += subBlockSize) {
        for (size_t j = 0; j < size; j += subBlockSize) {
            for (size_t k = 0; k < size; k += subBlockSize) {
                for (size_t ii = i; ii < i + subBlockSize && ii < size; ii++) {
                    for (size_t jj = j; jj < j + subBlockSize && jj < size; jj++) {
                        for (size_t kk = k; kk < k + subBlockSize && kk < size; kk++) {
                            matrixC[IND(ii, jj)] += matrixA[IND(ii,kk)] * matrixB[IND(kk,jj)];
                        }
                    }
                }
            }
        }
    }
}

