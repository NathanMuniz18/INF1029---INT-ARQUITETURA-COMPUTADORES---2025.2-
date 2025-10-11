#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h> 
#include "matrix_lib.h"
#include "timer.h"

/*
Para compilar:
gcc –std=c11 –mfma -o matrix_lib_test src/matrix_lib.c src/matrix_lib_test.c  src/timer.c
trab 3 gcc -std=c11 -pthread -mfma -o matrix_lib_test src/matrix_lib_test.c src/matrix_lib.c src/timer.c

Para executar:
./matrix_lib_test <escala> <A_linhas> <A_cols> <B_linhas> <B_cols> <arquivo_A> <arquivo_B> <arquivo_out1> <arquivo_out2>
./matrix_lib_test 5.0 1024 1024 1024 1024 4 test/matrix_1024x1024_1.dat test/matrix_1024x1024_2.dat test/result1.dat test/result2.dat
./matrix_lib_test 5.0 2048 2048 2048 2048 4 test/matrix_2048x2048_1.dat test/matrix_2048x2048_2.dat test/result1.dat test/result2.dat

*/

int main(int argc, char *argv[]) {
    if (argc != 11) {

        printf("Uso: %s <escala> <A_linhas> <A_cols> <B_linhas> <B_cols> <num_threads> <arquivo_A> <arquivo_B> <arquivo_out1> <arquivo_out2>\n", argv[0]);
        return 1; 
    } 

    // --- Conversão dos argumentos ---

    char *eptr;
    float scalar = strtof(argv[1], &eptr);
    unsigned long int A_h = strtoul(argv[2], &eptr, 10);
    unsigned long int A_w = strtoul(argv[3], &eptr, 10);
    unsigned long int B_h = strtoul(argv[4], &eptr, 10);
    unsigned long int B_w = strtoul(argv[5], &eptr, 10);
    int num_threads = strtol(argv[6], &eptr, 10);
    
    char *fileA = argv[7];
    char *fileB = argv[8];
    char *fileOut1 = argv[9];
    char *fileOut2 = argv[10];

    set_number_threads(num_threads);

    if (A_w != B_h) {
        printf("Erro: número de colunas de A deve ser igual ao número de linhas de B.\n");
        return 1;
    }

    // --- Alocação das matrizes ---
    struct matrix A = { A_h, A_w, malloc(A_h * A_w * sizeof(float)) };
    struct matrix B = { B_h, B_w, malloc(B_h * B_w * sizeof(float)) };
    struct matrix C = { A_h, B_w, malloc(A_h * B_w * sizeof(float)) };


    if (!A.rows || !B.rows || !C.rows) {
        printf("Erro na alocação de memória.\n");
        return 1;
    }



    // --- Leitura dos arquivos binários para A e B ---
    FILE *fa = fopen(fileA, "rb");
    if (!fa) { printf("Erro ao abrir arquivo A\n"); return 1; }

    unsigned long int totalA = A.height * A.width;
    for (unsigned long int i = 0; i < totalA; i += 8) {
        float buffer[8];
        fread(buffer, sizeof(float), 8, fa); // lê 8 floats do arquivo
        __m256 vec = _mm256_loadu_ps(buffer); // carrega no registrador AVX
        _mm256_storeu_ps(&A.rows[i], vec);    // armazena no array da matriz
    }
    fclose(fa);

    FILE *fb = fopen(fileB, "rb");
    if (!fb) { printf("Erro ao abrir arquivo B\n"); return 1; }

    unsigned long int totalB = B.height * B.width;
    for (unsigned long int i = 0; i < totalB; i += 8) {
        float buffer[8];
        fread(buffer, sizeof(float), 8, fb);
        __m256 vec = _mm256_loadu_ps(buffer);
        _mm256_storeu_ps(&B.rows[i], vec);
    }
    fclose(fb);



    //Inicializa C com zeros usando AVX
    __m256 zero_vec = _mm256_setzero_ps();
    unsigned long int totalC = C.height * C.width;
    for (unsigned long int i = 0; i < totalC; i += 8) {
        _mm256_storeu_ps(&C.rows[i], zero_vec);
    }

    struct timeval overall_t1, overall_t2, start, stop;
    gettimeofday(&overall_t1, NULL);

    // --- Scalar multiplication ---
    gettimeofday(&start, NULL);
    if (!scalar_matrix_mult(scalar, &A)) {
        printf("Erro na multiplicação escalar.\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    float time_scalar_mult = timedifference_msec(start, stop);

    FILE *fout1 = fopen(fileOut1, "wb");
    fwrite(A.rows, sizeof(float), A_h * A_w, fout1);
    fclose(fout1);

    printf("\nMatriz A (%lux%lu):\n", A.height, A.width);
    unsigned long int printedA = 0;
    for (unsigned long int i = 0; i < A.height && printedA < 256; ++i) {
        for (unsigned long int j = 0; j < A.width && printedA < 256; ++j) {
            printf("%8.2f ", A.rows[i * A.width + j]);
            printedA++;
        }
        printf("\n");
    }
    if (A.height * A.width > 256)
        printf("... (limite de 256 elementos atingido)\n");


    printf("\nMatriz B (%lux%lu):\n", B.height, B.width);
    unsigned long int printedB = 0;
    for (unsigned long int i = 0; i < B.height && printedB < 256; ++i) {
        for (unsigned long int j = 0; j < B.width && printedB < 256; ++j) {
            printf("%8.2f ", B.rows[i * B.width + j]);
            printedB++;
        }
        printf("\n");
    }
    if (B.height * B.width > 256)
        printf("... (limite de 256 elementos atingido)\n");


    // --- Matrix multiplication ---
    gettimeofday(&start, NULL);
    if (!matrix_matrix_mult(&A, &B, &C)) {
        printf("Erro na multiplicação de matrizes.\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    float time_matrix_mult = timedifference_msec(start, stop);

    FILE *fout2 = fopen(fileOut2, "wb");
    fwrite(C.rows, sizeof(float), C.height * C.width, fout2);
    fclose(fout2);

    printf("\nMatriz C (%lux%lu):\n", C.height, C.width);
    unsigned long int printedC = 0;
    for (unsigned long int i = 0; i < C.height && printedC < 256; ++i) {
        for (unsigned long int j = 0; j < C.width && printedC < 256; ++j) {
            printf("%8.2f ", C.rows[i * C.width + j]);
            printedC++;
        }
        printf("\n");
    }
    if (C.height * C.width > 256)
        printf("... (limite de 256 elementos atingido)\n");


    gettimeofday(&overall_t2, NULL);

    // --- Resultados ---
    printf("\n-------------------------------------------- Resultados: --------------------------------------------\n");
    printf("Tempo scalar_matrix_mult: %f ms\n", time_scalar_mult);
    printf("Tempo matrix_matrix_mult: %f ms\n", time_matrix_mult);
    printf("Tempo total: %f ms\n", timedifference_msec(overall_t1, overall_t2));

    free(A.rows);
    free(B.rows);
    free(C.rows);

    return 0;
}
