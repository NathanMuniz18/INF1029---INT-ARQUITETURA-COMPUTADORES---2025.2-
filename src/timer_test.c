#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

int main(int argc, char *argv[]) {
    struct timeval start, stop, overall_t1, overall_t2;
    unsigned long int i, VECTOR_SIZE, *vector;
    char *eptr = NULL;
    // Mark overall start time
    gettimeofday(&overall_t1, NULL);

    if (argc != 2) {
        printf("Usage: %s <vector_size>\n", argv[0]);
        return 0;
    }
    // Convert arguments
    VECTOR_SIZE = strtol(argv[1], &eptr, 10);
    // Allocate vector
    vector = (unsigned long int*)malloc(VECTOR_SIZE*sizeof(unsigned long int));
    if (vector == NULL) {
        printf("%s: vector allocation problem.", argv[0]);
        return 1;
    }
    // Mark init start time
    gettimeofday(&start, NULL);
    // Initialize vector
    for (i = 0; i < VECTOR_SIZE; ++i){
        vector[i] = i;
    }
    // Mark init stop time
    gettimeofday(&stop, NULL);
    // Show init exec time
    printf("Init time: %f ms\n", timedifference_msec(start, stop));
    // Mark reorder start time
    gettimeofday(&start, NULL);
    // Reorder vector
    for (i = 0; i < VECTOR_SIZE; ++i){
        vector[i] = VECTOR_SIZE - 1 - i;
    }
    // Mark reorder stop time
    gettimeofday(&stop, NULL);
    // Show reorder exec time
    printf("Reorder time: %f ms\n", timedifference_msec(start, stop));
    // Free vector
    free(vector);
    // Mark overall stop time
    gettimeofday(&overall_t2, NULL);
    // Show elapsed overall time
    printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));
    return 0;
}
