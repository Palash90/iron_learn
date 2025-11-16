// kernels/matrix_mul.cu
// Tiled matrix multiplication (row-major)
// C = A * B
// A: M x K
// B: K x N
// C: M x N
//
// Kernel name: matrix_mul
// Signature: (const float* A, const float* B, float* C, int M, int N, int K)

#include <cuda.h>

#define TILE 16

extern "C" __global__
void matrix_mul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread computes one element in the block sub-matrix
    int row = blockRow * TILE + threadIdx.y; // M dimension
    int col = blockCol * TILE + threadIdx.x; // N dimension

    // Accumulate value for C[row, col]
    float value = 0.0f;

    // Loop over tiles of A and B that are required to compute C element
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Shared memory for A and B tiles
        __shared__ float sA[TILE][TILE];
        __shared__ float sB[TILE][TILE];

        // Global indices to load
        int aRow = row;
        int aCol = t * TILE + threadIdx.x; // column in A (K dim)

        int bRow = t * TILE + threadIdx.y; // row in B (K dim)
        int bCol = col;

        // Load elements into shared memory (with bounds checks)
        if (aRow < M && aCol < K)
            sA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && bCol < N)
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE; ++k) {
            value += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

    