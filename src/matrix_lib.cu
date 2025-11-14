#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

static int g_threads_per_block = 256;
static int g_max_blocks_per_grid = 4096;


int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
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

    int device_max_threads = prop.maxThreadsPerBlock;      
    int device_max_blocks  = prop.maxGridSize[0];          

    if (threads_per_block <= 0 || max_blocks_per_grid <= 0) {
        g_threads_per_block = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    if (threads_per_block > device_max_threads || (long long)max_blocks_per_grid > (long long)device_max_blocks) {
        g_threads_per_block = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    g_threads_per_block = threads_per_block;
    g_max_blocks_per_grid = max_blocks_per_grid;
    return 1;
}

// CUDA kernels
__global__
void scalar_kernel(float scalar, float *d_rows, unsigned long long total) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long p = idx; p < total; p += stride) {
        d_rows[p] = d_rows[p] * scalar;
    }
}

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
        const float *Bcol_base = B + j; 
        for (unsigned int k = 0; k < A_w; ++k) {
            sum += Arow[k] * Bcol_base[(unsigned long long)k * B_w];
        }
        C[p] = sum;
    }
}

__global__
void matmul_row_kernel(const float *A_row, const float *B, float *C_row, unsigned int A_w, unsigned int B_w) {
    unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long col = j; col < B_w; col += stride) {
        float sum = 0.0f;
        const float *Bptr = B + col;
        for (unsigned int k = 0; k < A_w; ++k) {
            sum += A_row[k] * Bptr[(unsigned long long)k * B_w];
        }
        C_row[col] = sum;
    }
}


//Funcoes
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL) return 0;
    if (matrix->h_rows == NULL) return 0;

    unsigned long long total = (unsigned long long)matrix->height * (unsigned long long)matrix->width;

    // Caso FULL_ALLOC
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

        cerr = cudaMemcpy(matrix->h_rows, matrix->d_rows, (size_t)(total * sizeof(float)), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "scalar_matrix_mult: cudaMemcpy D2H error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        return 1;
    }

    // Caso PARTIAL_ALLOC
    if (matrix->alloc_mode == PARTIAL_ALLOC) {
        if (matrix->d_rows == NULL) return 0;
        unsigned long width = matrix->width;
        unsigned long long blocks = (width + (unsigned long long)g_threads_per_block - 1ULL) / (unsigned long long)g_threads_per_block;
        if (blocks > (unsigned long long)g_max_blocks_per_grid) blocks = (unsigned long long)g_max_blocks_per_grid;

        for (unsigned long i = 0; i < matrix->height; ++i) {
            float *host_row = matrix->h_rows + ((unsigned long long)i * width);
            cudaError_t cerr = cudaMemcpy(matrix->d_rows, host_row, (size_t)(width * sizeof(float)), cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "scalar_matrix_mult(partial): cudaMemcpy H2D error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
            scalar_kernel<<<(unsigned int)blocks, (unsigned int)g_threads_per_block>>>(scalar_value, matrix->d_rows, width);
            cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                fprintf(stderr, "scalar_matrix_mult(partial): kernel error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
            cerr = cudaMemcpy(host_row, matrix->d_rows, (size_t)(width * sizeof(float)), cudaMemcpyDeviceToHost);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "scalar_matrix_mult(partial): cudaMemcpy D2H error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
        }
        return 1;
    }

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

    // FULL_ALLOC
    if (matrixA->alloc_mode == FULL_ALLOC && matrixB->alloc_mode == FULL_ALLOC && matrixC->alloc_mode == FULL_ALLOC) {
        if (!matrixA->d_rows || !matrixB->d_rows || !matrixC->d_rows) return 0;

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

        cerr = cudaMemcpy(matrixC->h_rows, matrixC->d_rows, (size_t)(totalC * sizeof(float)), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "matrix_matrix_mult: cudaMemcpy D2H C error: %s\n", cudaGetErrorString(cerr));
            return 0;
        }

        return 1;
    }

    // --- PARTIAL_ALLOC
    if (matrixB->alloc_mode == FULL_ALLOC && matrixA->alloc_mode == PARTIAL_ALLOC && matrixC->alloc_mode == PARTIAL_ALLOC) {
        if (!matrixB->d_rows || !matrixA->d_rows || !matrixC->d_rows) {
            fprintf(stderr, "matrix_matrix_mult(partial): d_rows buffers missing\n");
            return 0;
        }

        unsigned long long blocks = (B_w + (unsigned long long)g_threads_per_block - 1ULL) / (unsigned long long)g_threads_per_block;
        if (blocks > (unsigned long long)g_max_blocks_per_grid) blocks = (unsigned long long)g_max_blocks_per_grid;

        for (unsigned int i = 0; i < A_h; ++i) {
            float *A_host_row = matrixA->h_rows + ((unsigned long long)i * A_w);
            cudaError_t cerr = cudaMemcpy(matrixA->d_rows, A_host_row, (size_t)(A_w * sizeof(float)), cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): cudaMemcpy A row H2D error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }

            cerr = cudaMemset(matrixC->d_rows, 0, (size_t)(B_w * sizeof(float)));
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): cudaMemset C row error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }

            matmul_row_kernel<<<(unsigned int)blocks, (unsigned int)g_threads_per_block>>>(
                matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, A_w, B_w
            );

            cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): kernel error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }

            float *C_host_row = matrixC->h_rows + ((unsigned long long)i * B_w);
            cerr = cudaMemcpy(C_host_row, matrixC->d_rows, (size_t)(B_w * sizeof(float)), cudaMemcpyDeviceToHost);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "matrix_matrix_mult(partial): cudaMemcpy C row D2H error: %s\n", cudaGetErrorString(cerr));
                return 0;
            }
        }

        return 1;
    }

    fprintf(stderr, "matrix_matrix_mult: unsupported alloc_mode configuration. A:%d B:%d C:%d\n",
            matrixA->alloc_mode, matrixB->alloc_mode, matrixC->alloc_mode);
    return 0;
}
