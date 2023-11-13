#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrixMultiplication.h"


int main(void) {
    const size_t matrixSize = 1024;

    // Allocate memory for matrices
    int *matrixA = (int *) malloc(matrixSize * matrixSize * sizeof(*matrixA));
    int *matrixB = (int *) malloc(matrixSize * matrixSize * sizeof(*matrixB));
    int *matrixC = (int *) malloc(matrixSize * matrixSize * sizeof(*matrixC));

    // error checking
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL) {
        printf("Error: memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    for (size_t i = 0; i < matrixSize * matrixSize; i++) {
        matrixA[i] = 1;
        matrixB[i] = 1;
    }

    clock_t start = clock();
    // Multiply matrices
    matrixMultiplication(matrixA, matrixB, matrixC, matrixSize);
    clock_t end = clock();
    printf("Time taken: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Validate the result
    for (size_t i = 0; i < matrixSize * matrixSize; i++) {
        if (matrixC[i] != matrixSize) {
            printf("Error: matrixC[%ld] = %d\n", i, matrixC[i]);
            return EXIT_FAILURE;
        }
    }

    // Free memory
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return EXIT_SUCCESS;    
}