#include "aux_functions.h"
#include <stdlib.h>

#define N 20000000

int main() {
    float *in = (float*)calloc(N, sizeof(float));
    float *out = (float*)calloc(N, sizeof(float));
    const float ref = 0.5f;

    for (int i=0; i<N; ++i) {
        in[i] = scale(i, N);
    }

    distanceArray(out, in, ref, N);

    // Release the heap memory after we are done using it
    free(in);
    free(out);
    return EXIT_SUCCESS;
}