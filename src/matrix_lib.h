#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *rows;
};

 struct ThreadArgs {
    float scalar;
    struct matrix *A;
    struct matrix *B;
    struct matrix *C;
    unsigned long start_row;
    unsigned long end_row;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);
void set_number_threads(int num_threads);
void* scalar_mult_worker(void *args);
void* matrix_mult_worker(void *args);


#endif
