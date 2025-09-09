#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *);

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

/*
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
}*/

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