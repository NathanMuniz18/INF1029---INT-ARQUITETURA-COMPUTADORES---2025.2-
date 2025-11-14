#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

static int g_threads_per_block = 256;
static int g_max_blocks_per_grid = 4096;

// ----------------------------------------------
// set_grid_size
// ----------------------------------------------
int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        // não conseguimos descobrir a device; aplicar defaults e sinalizar erro
        g_threads_per_block = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        g_threads_per_block = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    int device_max_threads = prop.maxThreadsPerBlock;      // tipicamente 1024
    int device_max_blocks  = prop.maxGridSize[0];          // limite no eixo X (ex: 65535 ou maior)

    if (threads_per_block <= 0 || max_blocks_per_grid <= 0) {
        // inválido -> usa defaults
        g_threads_per_block = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    if (threads_per_block > device_max_threads || (long long)max_blocks_per_grid > (long long)device_max_blocks) {
        // extrapolou limites do device -> defaults e erro
        g_threads_per_block = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    // aceitável
    g_threads_per_block = threads_per_block;
    g_max_blocks_per_grid = max_blocks_per_grid;
    return 1;
}

// ----------------------------------------------
// CUDA kernels
// ----------------------------------------------

// kernel para multiplicação escalar sobre um vetor linear de floats (total elementos)
__global__
void scalar_kernel(float scalar, float *d_rows, unsigned long long total) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long p = idx; p < total; p += stride) {
        d_rows[p] = d_rows[p] * scalar;
    }
}

// kernel para multiplicação matricial FULL (cada thread calcula 1 ou mais elementos de C)
// A: A_h x A_w, B: A_w x B_w, C: A_h x B_w
__global__
void matmul_full_kernel(const float *A, const float *B, float *C,
                        unsigned int A_h, unsigned int A_w, unsigned int B_w) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long totalC = (unsigned long long)A_h * (unsigned long long)B_w;

    for (unsigned long long p = idx; p < totalC; p += stride) {
        unsigned int i = (unsigned int)(p / B_w);
        unsigned int j = (unsigned int)(p % B_w);
        float sum = 0.0f;
        const float *Arow = A + ((unsigned long long)i * A_w);
        const float *Bcol_base = B + j; // B[k * B_w + j] -> base + k*B_w
        for (unsigned int k = 0; k < A_w; ++k) {
            sum += Arow[k] * Bcol_base[(unsigned long long)k * B_w];
        }
        C[p] = sum;
    }
}

// kernel para computar UMA linha de C dada uma linha de A e B inteiro em device
// A_row: vetor length A_w, B: A_w x B_w, C_row: vetor length B_w
__global__
void matmul_row_kernel(const float *A_row, const float *B, float *C_row, unsigned int A_w, unsigned int B_w) {
    unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long col = j; col < B_w; col += stride) {
        float sum = 0.0f;
        // soma em k: A_row[k] * B[k*B_w + col]
        const float *Bptr = B + col;
        for (unsigned int k = 0; k < A_w; ++k) {
            sum += A_row[k] * Bptr[(unsigned long long)k * B_w];
        }
        C_row[col] = sum;
    }
}

// ----------------------------------------------
// Wrappers host: scalar_matrix_mult e matrix_matrix_mult
// ----------------------------------------------

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL) return 0;
    if (matrix->h_rows == NULL) return 0;

    unsigned long long total = (unsigned long long)matrix->height * (unsigned long long)matrix->width;

    // Caso FULL_ALLOC: assumimos que matrix->d_rows aponta para todos os elementos já alocados no device
    if (matrix->alloc_mode == FULL_ALLOC) {
        if (matrix->d_rows == NULL) return 0;

        unsigned long long blocks = (total + (unsigned long long)g_threads_per_block - 1ULL) / (unsigned long long)g_threads_per_block;
        if (blocks > (unsigned long long)g_max_blocks_per_grid) blocks = (unsigned long long)g_max_blocks_per_grid;

        scalar_kernel<<<(unsigned int)blocks, (unsigned int)g_threads_per_block>>>(scalar_value, matrix->d_rows, total);
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "scalar_matrix_mult: kernel error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        // copia de volta ao host
        cerr = cudaMemcpy(matrix->h_rows, matrix->d_rows, (size_t)(total * sizeof(float)), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "scalar_matrix_mult: cudaMemcpy D2H error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        return 1;
    }

    // Caso PARTIAL_ALLOC: assumimos que matrix->d_rows é um buffer no device suficiente para UMA linha (width floats).
    // Iteramos por cada linha: copia host->device (linha), roda kernel sobre a linha (total = width), copia device->host
    if (matrix->alloc_mode == PARTIAL_ALLOC) {
        if (matrix->d_rows == NULL) return 0;
        unsigned long width = matrix->width;
        unsigned long long blocks = (width + (unsigned long long)g_threads_per_block - 1ULL) / (unsigned long long)g_threads_per_block;
        if (blocks > (unsigned long long)g_max_blocks_per_grid) blocks = (unsigned long long)g_max_blocks_per_grid;

        for (unsigned long i = 0; i < matrix->height; ++i) {
            float *host_row = matrix->h_rows + ((unsigned long long)i * width);
            // copiar linha host -> device
            cudaError_t cerr = cudaMemcpy(matrix->d_rows, host_row, (size_t)(width * sizeof(float)), cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "scalar_matrix_mult(partial): cudaMemcpy H2D error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
            // executar kernel sobre a linha (total = width)
            scalar_kernel<<<(unsigned int)blocks, (unsigned int)g_threads_per_block>>>(scalar_value, matrix->d_rows, width);
            cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                fprintf(stderr, "scalar_matrix_mult(partial): kernel error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
            // copiar de volta device -> host
            cerr = cudaMemcpy(host_row, matrix->d_rows, (size_t)(width * sizeof(float)), cudaMemcpyDeviceToHost);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "scalar_matrix_mult(partial): cudaMemcpy D2H error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
        }
        return 1;
    }

    // Caso sem flag conhecida
    return 0;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL) return 0;
    if (matrixA->width != matrixB->height) return 0;
    if (matrixC->height != matrixA->height || matrixC->width != matrixB->width) return 0;

    unsigned int A_h = (unsigned int)matrixA->height;
    unsigned int A_w = (unsigned int)matrixA->width;
    unsigned int B_w = (unsigned int)matrixB->width;
    unsigned long long totalC = (unsigned long long)A_h * (unsigned long long)B_w;

    // --- FULL_ALLOC: assume d_rows completos para A, B e C
    if (matrixA->alloc_mode == FULL_ALLOC && matrixB->alloc_mode == FULL_ALLOC && matrixC->alloc_mode == FULL_ALLOC) {
        if (!matrixA->d_rows || !matrixB->d_rows || !matrixC->d_rows) return 0;

        // opcional: zerar C no device antes (se não estiver zerado)
        cudaError_t cerr = cudaMemset(matrixC->d_rows, 0, (size_t)(totalC * sizeof(float)));
        if (cerr != cudaSuccess) {
            fprintf(stderr, "matrix_matrix_mult: cudaMemset error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        unsigned long long blocks = (totalC + (unsigned long long)g_threads_per_block - 1ULL) / (unsigned long long)g_threads_per_block;
        if (blocks > (unsigned long long)g_max_blocks_per_grid) blocks = (unsigned long long)g_max_blocks_per_grid;

        matmul_full_kernel<<<(unsigned int)blocks, (unsigned int)g_threads_per_block>>>(
            matrixA->d_rows, matrixB->d_rows, matrixC->d_rows,
            A_h, A_w, B_w
        );

        cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "matrix_matrix_mult: kernel error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        // copia C de volta ao host
        cerr = cudaMemcpy(matrixC->h_rows, matrixC->d_rows, (size_t)(totalC * sizeof(float)), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "matrix_matrix_mult: cudaMemcpy D2H C error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        return 1;
    }

    // --- PARTIAL_ALLOC: B completo no device; A and C tem apenas buffers de 1 linha em d_rows
    // Convenção exigida: matrixB->d_rows possui B_h * B_w floats; matrixA->d_rows e matrixC->d_rows apontam
    // para buffers com capacidade para 'width' (uma linha).
    if (matrixB->alloc_mode == FULL_ALLOC && matrixA->alloc_mode == PARTIAL_ALLOC && matrixC->alloc_mode == PARTIAL_ALLOC) {
        if (!matrixB->d_rows || !matrixA->d_rows || !matrixC->d_rows) {
            fprintf(stderr, "matrix_matrix_mult(partial): d_rows buffers missing\n");
            return 0;
        }

        // Para cada linha i de A:
        // 1) copia A.h_rows[i*width .. ] -> A.d_rows (buffer de 1 linha)
        // 2) zera C.d_rows (buffer de 1 linha)
        // 3) executar kernel matmul_row_kernel(A.d_rows, B.d_rows, C.d_rows, A_w, B_w)
        // 4) copiar C.d_rows -> C.h_rows[i*B_w .. ]
        unsigned long long blocks = (B_w + (unsigned long long)g_threads_per_block - 1ULL) / (unsigned long long)g_threads_per_block;
        if (blocks > (unsigned long long)g_max_blocks_per_grid) blocks = (unsigned long long)g_max_blocks_per_grid;

        for (unsigned int i = 0; i < A_h; ++i) {
            // copiar A row host->device
            float *A_host_row = matrixA->h_rows + ((unsigned long long)i * A_w);
            cudaError_t cerr = cudaMemcpy(matrixA->d_rows, A_host_row, (size_t)(A_w * sizeof(float)), cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): cudaMemcpy A row H2D error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }

            // zera C row no device
            cerr = cudaMemset(matrixC->d_rows, 0, (size_t)(B_w * sizeof(float)));
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): cudaMemset C row error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }

            // calcular a linha
            matmul_row_kernel<<<(unsigned int)blocks, (unsigned int)g_threads_per_block>>>(
                matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, A_w, B_w
            );

            cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): kernel error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }

            // copiar C row device->host
            float *C_host_row = matrixC->h_rows + ((unsigned long long)i * B_w);
            cerr = cudaMemcpy(C_host_row, matrixC->d_rows, (size_t)(B_w * sizeof(float)), cudaMemcpyDeviceToHost);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): cudaMemcpy C row D2H error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
        }

        return 1;
    }

    // Se chegamos aqui -> configuração de alocação não suportada pela implementação
    fprintf(stderr, "matrix_matrix_mult: unsupported alloc_mode configuration. A:%d B:%d C:%d\n",
            matrixA->alloc_mode, matrixB->alloc_mode, matrixC->alloc_mode);
    return 0;
}
