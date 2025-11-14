#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

// Tipos de alocação suportados pelo Trabalho 4
#define FULL_ALLOC     1
#define PARTIAL_ALLOC  0

// Estrutura da matriz usada no Trabalho 4
struct matrix {
    unsigned long int height;   // número de linhas
    unsigned long int width;    // número de colunas
    float *h_rows;              // memória no host (CPU)
    float *d_rows;              // memória no device (GPU)
    int alloc_mode;             // FULL_ALLOC ou PARTIAL_ALLOC
};


int set_grid_size(int threads_per_block, int max_blocks_per_grid);

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *matrixA,
                       struct matrix *matrixB,
                       struct matrix *matrixC);

#endif
