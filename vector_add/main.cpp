#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vectorAdd.h"

int main(int argc, char **argv) {
    int n = 100000000;
    int *a = (int *) malloc(n * sizeof(int));
    int *b = (int *) malloc(n * sizeof(int));
    int *c = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    clock_t start = clock();
    vectorAdd(a, b, c, n);
    clock_t end = clock();

    for (int i = 0; i < n; i++) {
        if (c[i] != 2 * i) {
            printf("Error: c[%d] = %d\n", i, c[i]);
        }
    }

    // Compute time in seconds between clock count readings.
    double time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Time: %f\n", time);
    printf("\n");

    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}