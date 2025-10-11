#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> 
#include "matrix_lib.h"
#include <pthread.h>
static int number_of_threads = 1;

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);
void set_number_threads(int num_threads);
void* scalar_mult_worker(void *args);
void* matrix_mult_worker(void *args);

void set_number_threads(int num_threads) {
    if (num_threads > 0) {
        number_of_threads = num_threads;
    } else {
        number_of_threads = 1;
    }
}

void* scalar_mult_worker(void *args) {
    struct ThreadArgs *thread_args = (struct ThreadArgs *)args;
    float scalar = thread_args->scalar;
    struct matrix *A = thread_args->A;
    unsigned long start_row = thread_args->start_row;
    unsigned long end_row = thread_args->end_row;

    // 1. Crie um vetor AVX onde todas as 8 posições contêm o valor escalar
    __m256 scalar_vec = _mm256_set1_ps(scalar);

    // 2. O laço externo percorre as linhas designadas para esta thread 
    for (unsigned long i = start_row; i < end_row; i++) {
        unsigned long row_offset = i * A->width;
        
        // 3. O laço interno agora avança de 8 em 8 colunas
        for (unsigned long j = 0; j < A->width; j += 8) {
            // Carrega 8 floats da matriz para um vetor AVX
            __m256 matrix_vec = _mm256_loadu_ps(&A->rows[row_offset + j]);
            
            // Multiplica os 8 floats da matriz pelo escalar (tudo em uma instrução)
            matrix_vec = _mm256_mul_ps(matrix_vec, scalar_vec);
            
            // Armazena os 8 resultados de volta na matriz
            _mm256_storeu_ps(&A->rows[row_offset + j], matrix_vec);
        }
    }
    return NULL;
}
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    if (matrix == NULL || matrix->rows == NULL) return 0;

    pthread_t threads[number_of_threads];
    struct ThreadArgs thread_args[number_of_threads];

    unsigned long rows_per_thread = matrix->height / number_of_threads;
    unsigned long remaining_rows = matrix->height % number_of_threads;

    unsigned long current_row = 0;
    for (int i = 0; i < number_of_threads; i++) {
        thread_args[i].scalar = scalar_value;
        thread_args[i].A = matrix;
        thread_args[i].start_row = current_row;
        thread_args[i].end_row = current_row + rows_per_thread + (i < remaining_rows ? 1 : 0);
        current_row = thread_args[i].end_row;

        pthread_create(&threads[i], NULL, scalar_mult_worker, &thread_args[i]);
    }

    for (int i = 0; i < number_of_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 1;
}

void* matrix_mult_worker(void *args){
    struct ThreadArgs *thread_args = (struct ThreadArgs *)args;
    float scalar = thread_args->scalar;
    struct matrix *A = thread_args->A;
    struct matrix *B = thread_args->B;
    struct matrix *C = thread_args->C;
    unsigned long start_row = thread_args->start_row;
    unsigned long end_row = thread_args->end_row;

    unsigned long int A_w = A->width;
    unsigned long int B_w = B->width;
    
    for (unsigned long int i = start_row; i < end_row; i++) {
        unsigned long int a_row_offset = i * A_w;
        unsigned long int c_row_offset = i * B_w;

        for (unsigned long int k = 0; k < A_w; k++) {
            float a_scalar = A->rows[a_row_offset + k];
            __m256 a_vec = _mm256_set1_ps(a_scalar);

            unsigned long int b_row_offset = k * B_w;

            for (unsigned long int j = 0; j < B_w; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B->rows[b_row_offset + j]);
                __m256 c_vec = _mm256_loadu_ps(&C->rows[c_row_offset + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C->rows[c_row_offset + j], c_vec);
            }
        }
    }
    return NULL;
}
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC){
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL) return 0;
    if (matrixA->width != matrixB->height) return 0;
    if (matrixC->height != matrixA->height || matrixC->width != matrixB->width) return 0;

    pthread_t threads[number_of_threads];
    struct ThreadArgs thread_args[number_of_threads];

    unsigned long rows_per_thread = matrixA->height / number_of_threads;
    unsigned long remaining_rows = matrixA->height % number_of_threads;

    unsigned long current_row = 0;
    for (int i = 0; i < number_of_threads; i++) {
        thread_args[i].A = matrixA;
        thread_args[i].B = matrixB;
        thread_args[i].C = matrixC;
        thread_args[i].start_row = current_row;
        thread_args[i].end_row = current_row + rows_per_thread + (i < remaining_rows ? 1 : 0);
        current_row = thread_args[i].end_row;

        pthread_create(&threads[i], NULL, matrix_mult_worker, &thread_args[i]);
    }

    for (int i = 0; i < number_of_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 1;
}

//versao otimizada 2 com AVX
// int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
//     if (matrix == NULL || matrix->rows == NULL) return 0;

//     unsigned long int total = matrix->height * matrix->width;
//     __m256 scalar_vec = _mm256_set1_ps(scalar_value);

//     for (unsigned long int i = 0; i < total; i += 8) {
//         __m256 v = _mm256_loadu_ps(&matrix->rows[i]);
//         v = _mm256_mul_ps(v, scalar_vec);
//         _mm256_storeu_ps(&matrix->rows[i], v);
//     }
//     return 1;
// }

// int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
//     if (matrixA == NULL || matrixB == NULL || matrixC == NULL) return 0;
//     if (matrixA->width != matrixB->height) return 0;
//     if (matrixC->height != matrixA->height || matrixC->width != matrixB->width) return 0;

//     unsigned long int A_h = matrixA->height;
//     unsigned long int A_w = matrixA->width;
//     unsigned long int B_w = matrixB->width;

//     for (unsigned long int i = 0; i < A_h; i++) {
//         unsigned long int a_row_offset = i * A_w;
//         unsigned long int c_row_offset = i * B_w;

//         for (unsigned long int k = 0; k < A_w; k++) {
//             float a_scalar = matrixA->rows[a_row_offset + k];
//             __m256 a_vec = _mm256_set1_ps(a_scalar);

//             unsigned long int b_row_offset = k * B_w;

//             for (unsigned long int j = 0; j < B_w; j += 8) {
//                 __m256 b_vec = _mm256_loadu_ps(&matrixB->rows[b_row_offset + j]);
//                 __m256 c_vec = _mm256_loadu_ps(&matrixC->rows[c_row_offset + j]);
//                 c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
//                 _mm256_storeu_ps(&matrixC->rows[c_row_offset + j], c_vec);
//             }
//         }
//     }
//     return 1;
// }

/*
//Versao Otimizada 1
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0; // erro
    }

    unsigned long int total = matrix->height * matrix->width;

    for (unsigned long int i = 0; i < total; i++) {
        matrix->rows[i] *= scalar_value;
    }

    return 1; // sucesso
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL) {
        return 0; // erro
    }

    if (matrixA->width != matrixB->height) {
        printf("matrixA width: %lu, matrixB height: %lu\n", matrixA->width, matrixB->height);
        return 0; // dimensões incompatíveis
    }

    if (matrixC->height != matrixA->height || matrixC->width != matrixB->width) {
        printf("matrixC height: %lu, expected: %lu\n", matrixC->height, matrixA->height);
        printf("matrixC width: %lu, expected: %lu\n", matrixC->width, matrixB->width);
        return 0; // matriz C com dimensões erradas
    }

    for (unsigned int i = 0; i < matrixA->height; i++) {   // percorre as linhas de A
        for (unsigned int k = 0; k < matrixA->width; k++) { // percorre cada elemento da linha i de A e da coluna j de B
            float a = matrixA->rows[i * matrixA->width + k];
            for(unsigned int j = 0; j < matrixB->width; j++)
            {
                float b = matrixB->rows[k * matrixB->width + j];
                matrixC->rows[i * matrixC->width + j] += a * b; // acumula o resultado em C[i][j]
            }
        }
    }
    return 1;
}

// Versão inicial
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL) {
        return 0; // erro
    }

    if (matrixA->width != matrixB->height) {
        return 0; // dimensões incompatíveis
    }

    if (matrixC->height != matrixA->height || matrixC->width != matrixB->width) {
        return 0; // matriz C com dimensões erradas
    }

    for (unsigned long int i = 0; i < matrixA->height; i++) {   // percorre as linhas de A
        for (unsigned long int j = 0; j < matrixB->width; j++) { // percorre as colunas de B
            float sum = 0.0;
            for (unsigned long int k = 0; k < matrixA->width; k++) { // percorre cada elemento da linha i de A e da coluna j de B
                float a = matrixA->rows[i * matrixA->width + k];
                float b = matrixB->rows[k * matrixB->width + j];
                sum += a * b;
            }
            matrixC->rows[i * matrixC->width + j] = sum; // resultado em C[i][j]
        }
    }
    return 1; // sucesso
}
*/  

