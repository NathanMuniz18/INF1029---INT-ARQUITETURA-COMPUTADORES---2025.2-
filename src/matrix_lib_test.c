#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "matrix_lib.h"
#include "timer.h"

/*
Para compilar:
gcc -o matrix_lib_test src/matrix_lib.c src/matrix_lib_test.c  src/timer.c

Para executar:
./matrix_lib_test <escala> <A_linhas> <A_cols> <B_linhas> <B_cols> <arquivo_A> <arquivo_B> <arquivo_out1> <arquivo_out2>
./matrix_lib_test 5.0 3 4 4 3 test/matrix_3x4.dat test/matrix_4x3.dat test/result1.dat test/result2.dat
*/

int main(int argc, char *argv[]) {
    if (argc != 10) {
        printf("Uso: %s <escala> <A_linhas> <A_cols> <B_linhas> <B_cols> <arquivo_A> <arquivo_B> <arquivo_out1> <arquivo_out2>\n", argv[0]);
        return 1;
    }

    // --- Conversão dos argumentos ---
    char *eptr;
    float scalar = strtof(argv[1], &eptr);
    unsigned long int A_h = strtoul(argv[2], &eptr, 10);
    unsigned long int A_w = strtoul(argv[3], &eptr, 10);
    unsigned long int B_h = strtoul(argv[4], &eptr, 10);
    unsigned long int B_w = strtoul(argv[5], &eptr, 10);

    char *fileA = argv[6];
    char *fileB = argv[7];
    char *fileOut1 = argv[8];
    char *fileOut2 = argv[9];

    // --- Verificação das dimensões ---
    if (A_w != B_h) {
        printf("Erro: número de colunas de A deve ser igual ao número de linhas de B.\n");
        return 1;
    }

    // --- Alocação das matrizes ---
    struct matrix A = { A_h, A_w, malloc(A_h * A_w * sizeof(float)) };
    struct matrix B = { B_h, B_w, malloc(B_h * B_w * sizeof(float)) };
    struct matrix C = { A_h, B_w, calloc(A_h * B_w, sizeof(float)) };

    if (!A.rows || !B.rows || !C.rows) {
        printf("Erro na alocação de memória.\n");
        return 1;
    }

    // --- Leitura dos arquivos binários ---
    FILE *fa = fopen(fileA, "rb");
    FILE *fb = fopen(fileB, "rb");
    if (!fa || !fb) {
        printf("Erro ao abrir arquivos de entrada.\n");
        return 1;
    }
    fread(A.rows, sizeof(float), A_h * A_w, fa);
    fread(B.rows, sizeof(float), B_h * B_w, fb);
    fclose(fa);
    fclose(fb);

    struct timeval overall_t1, overall_t2, start, stop;
    gettimeofday(&overall_t1, NULL);

    // --- Scalar multiplication ---
    gettimeofday(&start, NULL);
    if (!scalar_matrix_mult(scalar, &A)) {
        printf("Erro na multiplicação escalar.\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("Tempo scalar_matrix_mult: %f ms\n", timedifference_msec(start, stop));

    // --- Salvando resultado de A ---
    FILE *fout1 = fopen(fileOut1, "wb");
    fwrite(A.rows, sizeof(float), A_h * A_w, fout1);
    fclose(fout1);

    // --- Matrix multiplication ---
    gettimeofday(&start, NULL);
    if (!matrix_matrix_mult(&A, &B, &C)) {
        printf("Erro na multiplicação de matrizes.\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("Tempo matrix_matrix_mult: %f ms\n", timedifference_msec(start, stop));

    // --- Salvando resultado de C ---
    FILE *fout2 = fopen(fileOut2, "wb");
    fwrite(C.rows, sizeof(float), C.height * C.width, fout2);
    fclose(fout2);

    // --- Tempo total ---
    gettimeofday(&overall_t2, NULL);
    printf("Tempo total: %f ms\n", timedifference_msec(overall_t1, overall_t2));

    // --- Liberação ---
    free(A.rows);
    free(B.rows);
    free(C.rows);

    return 0;
}
