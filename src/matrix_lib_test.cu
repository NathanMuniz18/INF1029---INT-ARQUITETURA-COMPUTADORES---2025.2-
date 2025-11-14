#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"
#include "timer.h"


/*
Para compilar:
    nvcc -O3 -o matrix_lib_test src/matrix_lib_test.cu src/matrix_lib.cu src/timer.cu

Para executar:

./matrix_lib_test <escala> <A_linhas> <A_cols> <B_linhas> <B_cols> <threads_por_bloco> <max_blocos_por_grid> <max_gpu_mem_MiB> \
                      <arquivo_A> <arquivo_B> <arquivo_out1> <arquivo_out2>

Exemplo (1024x1024):
./matrix_lib_test 5.0 1024 1024 1024 1024 256 4096 4096 test/matrix_1024x1024_1.dat test/matrix_1024x1024_2.dat test/result1.dat test/result2.dat

Exemplo (2048x2048):
    ./matrix_lib_test 3.0 2048 2048 2048 2048 256 4096 4096 test/matrix_2048x2048_1.dat test/matrix_2048x2048_2.dat test/result1.dat test/result2.dat
*/


static void print_matrix(const char *name, float *rows,
                         unsigned long h, unsigned long w)
{
    printf("\n%s (%lux%lu):\n", name, h, w);
    unsigned long count = 0;
    for (unsigned long i = 0; i < h && count < 256; i++) {
        for (unsigned long j = 0; j < w && count < 256; j++) {
            printf("%8.2f ", rows[i*w + j]);
            count++;
        }
        printf("\n");
    }
    if (h * w > 256)
        printf("... (limite de 256 elementos atingido)\n");
}

int main(int argc, char *argv[])
{
    if (argc != 13) {
        printf("Uso:\n");
        printf("%s <escala> <A_h> <A_w> <B_h> <B_w> "
               "<threads_per_block> <max_blocks> <max_gpu_mem_MiB> "
               "<arquivo_A> <arquivo_B> <arquivo_out1> <arquivo_out2>\n",
                argv[0]);
        return 1;
    }

    char *ep;
    float scalar = strtof(argv[1], &ep);
    unsigned long A_h = strtoul(argv[2], &ep, 10);
    unsigned long A_w = strtoul(argv[3], &ep, 10);
    unsigned long B_h = strtoul(argv[4], &ep, 10);
    unsigned long B_w = strtoul(argv[5], &ep, 10);

    int threads_per_block = strtol(argv[6], &ep, 10);
    int max_blocks         = strtol(argv[7], &ep, 10);
    unsigned long max_gpu_mem_MiB = strtoul(argv[8], &ep, 10);

    char *fileA = argv[9];
    char *fileB = argv[10];
    char *fileOut1 = argv[11];
    char *fileOut2 = argv[12];

    if (A_w != B_h) {
        printf("Dimensões incompatíveis: A_w != B_h\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // 1. Alocação HOST
    // -------------------------------------------------------------------------
    struct matrix A, B, C;

    A.height = A_h; A.width = A_w;
    B.height = B_h; B.width = B_w;
    C.height = A_h; C.width = B_w;

    unsigned long sizeA = A_h * A_w * sizeof(float);
    unsigned long sizeB = B_h * B_w * sizeof(float);
    unsigned long sizeC = A_h * B_w * sizeof(float);

    A.h_rows = (float*)malloc(sizeA);
    B.h_rows = (float*)malloc(sizeB);
    C.h_rows = (float*)malloc(sizeC);

    if (!A.h_rows || !B.h_rows || !C.h_rows) {
        printf("Erro: malloc no host.\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // 2. LER arquivos binários em A.h_rows e B.h_rows
    // -------------------------------------------------------------------------
    FILE *fa = fopen(fileA, "rb");
    if (!fa) { printf("Erro ao abrir %s\n", fileA); return 1; }
    fread(A.h_rows, sizeof(float), A_h * A_w, fa);
    fclose(fa);

    FILE *fb = fopen(fileB, "rb");
    if (!fb) { printf("Erro ao abrir %s\n", fileB); return 1; }
    fread(B.h_rows, sizeof(float), B_h * B_w, fb);
    fclose(fb);

    // Iniciar C com zeros
    for (unsigned long i = 0; i < A_h * B_w; i++)
        C.h_rows[i] = 0.0f;

    // -------------------------------------------------------------------------
    // 3. Tentar FULL_ALLOC
    // -------------------------------------------------------------------------
    unsigned long needed = sizeA + sizeB + sizeC;
    unsigned long max_bytes = max_gpu_mem_MiB * 1024UL * 1024UL;

    A.alloc_mode = PARTIAL_ALLOC;
    B.alloc_mode = PARTIAL_ALLOC;
    C.alloc_mode = PARTIAL_ALLOC;

    if (needed <= max_bytes) {
        if (cudaMalloc((void**)&A.d_rows, sizeA) == cudaSuccess &&
            cudaMalloc((void**)&B.d_rows, sizeB) == cudaSuccess &&
            cudaMalloc((void**)&C.d_rows, sizeC) == cudaSuccess)
        {
            A.alloc_mode = FULL_ALLOC;
            B.alloc_mode = FULL_ALLOC;
            C.alloc_mode = FULL_ALLOC;

            cudaMemcpy(A.d_rows, A.h_rows, sizeA, cudaMemcpyHostToDevice);
            cudaMemcpy(B.d_rows, B.h_rows, sizeB, cudaMemcpyHostToDevice);
            cudaMemset(C.d_rows, 0, sizeC);
        }
        else {
            // fallback — libera e continua com partial
            cudaFree(A.d_rows);
            cudaFree(B.d_rows);
            cudaFree(C.d_rows);
        }
    }

    // -------------------------------------------------------------------------
    // 4. PARTIAL_ALLOC (fallback)
    // -------------------------------------------------------------------------
    if (A.alloc_mode == PARTIAL_ALLOC ||
        B.alloc_mode == PARTIAL_ALLOC ||
        C.alloc_mode == PARTIAL_ALLOC)
    {
        // Em PARTIAL, B precisa ser FULL_ALLOC
        unsigned long sizeBfull = sizeB;
        if (cudaMalloc((void**)&B.d_rows, sizeBfull) != cudaSuccess) {
            printf("Erro: não foi possível alocar B na GPU.\n");
            return 1;
        }
        B.alloc_mode = FULL_ALLOC;
        cudaMemcpy(B.d_rows, B.h_rows, sizeBfull, cudaMemcpyHostToDevice);

        // A.d_rows = buffer de 1 linha
        if (cudaMalloc((void**)&A.d_rows, A_w * sizeof(float)) != cudaSuccess) {
            printf("Erro: não foi possível alocar buffer A.\n");
            return 1;
        }
        A.alloc_mode = PARTIAL_ALLOC;

        // C.d_rows = buffer de 1 linha
        if (cudaMalloc((void**)&C.d_rows, B_w * sizeof(float)) != cudaSuccess) {
            printf("Erro: não foi possível alocar buffer C.\n");
            return 1;
        }
        C.alloc_mode = PARTIAL_ALLOC;
    }

    // -------------------------------------------------------------------------
    // 5. GRID SETUP
    // -------------------------------------------------------------------------
    set_grid_size(threads_per_block, max_blocks);

    // -------------------------------------------------------------------------
    // 6. Medição
    // -------------------------------------------------------------------------
    struct timeval t0, t1, t2, t3, total0, total1;
    gettimeofday(&total0, NULL);

    // -------------------------------------------------------------------------
    // 7. scalar_matrix_mult
    // -------------------------------------------------------------------------
    gettimeofday(&t0, NULL);
    if (!scalar_matrix_mult(scalar, &A)) {
        printf("Erro em scalar_matrix_mult\n");
        return 1;
    }
    gettimeofday(&t1, NULL);
    float t_scalar = timedifference_msec(t0, t1);

    // salvar resultado de A
    FILE *f1 = fopen(fileOut1, "wb");
    fwrite(A.h_rows, sizeof(float), A_h*A_w, f1);
    fclose(f1);

    // -------------------------------------------------------------------------
    // 8. matrix_matrix_mult
    // -------------------------------------------------------------------------
    gettimeofday(&t2, NULL);
    if (!matrix_matrix_mult(&A, &B, &C)) {
        printf("Erro em matrix_matrix_mult\n");
        return 1;
    }
    gettimeofday(&t3, NULL);
    float t_matmul = timedifference_msec(t2, t3);

    // salvar C
    FILE *f2 = fopen(fileOut2, "wb");
    fwrite(C.h_rows, sizeof(float), A_h * B_w, f2);
    fclose(f2);

    gettimeofday(&total1, NULL);

    // -------------------------------------------------------------------------
    // 9. Prints (até 256 elementos)
    // -------------------------------------------------------------------------
    print_matrix("Matriz A (após scalar)", A.h_rows, A_h, A_w);
    print_matrix("Matriz B", B.h_rows, B_h, B_w);
    print_matrix("Matriz C = A × B", C.h_rows, A_h, B_w);

    // -------------------------------------------------------------------------
    // 10. Infos finais
    // -------------------------------------------------------------------------
    printf("\n-----------------------------------------\n");
    printf("Tempo scalar_matrix_mult: %f ms\n", t_scalar);
    printf("Tempo matrix_matrix_mult: %f ms\n", t_matmul);
    printf("Tempo total: %f ms\n", timedifference_msec(total0, total1));
    printf("-----------------------------------------\n");

    // -------------------------------------------------------------------------
    // 11. Free
    // -------------------------------------------------------------------------
    cudaFree(A.d_rows);
    cudaFree(B.d_rows);
    cudaFree(C.d_rows);
    free(A.h_rows);
    free(B.h_rows);
    free(C.h_rows);

    return 0;
}
