#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
/* Kernels for Modular approach*/
extern "C" __global__ void fill_value(float *out, int n, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = value;
    }
}

extern "C" __global__ void vector_add(const float *a, const float *b, float *out, int n, int sub)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (sub == 1)
        {
            out[idx] = a[idx] - b[idx];
        }
        else
        {
            out[idx] = a[idx] + b[idx];
        }
    }
}

extern "C" __global__ void element_op(const float *s, float *r, int n, int op, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        switch (op)
        {
        case 0:
            r[idx] = exp(s[idx]);
            break;
        case 1:
            r[idx] = s[idx] * scale;
            break;
        case 2:
            r[idx] = sin(s[idx]);
            break;
        case 3:
            r[idx] = cos(s[idx]);
            break;
        case 4:
            r[idx] = tan(s[idx]);
            break;
        case 5:
            r[idx] = tanh(s[idx]);
            break;
        case 6:
            if (s[idx] >= 0.0f)
            {
                r[idx] = 1.0f / (1.0f + exp(-s[idx]));
            }
            else
            {
                r[idx] = exp(s[idx]) / (1.0f + exp(s[idx]));
            }
            break;
        case 7:
            r[idx] = log10(s[idx]);
            break;
        case 8:
            r[idx] = log(s[idx]);
            break;
        default:
            break;
        }
    }
}

extern "C" __global__ void compare_memory(const float *a, const float *b, size_t size, int *result)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float EPSILON = 1e-9;

    if (idx < size)
    {
        float diff = abs(a[idx] - b[idx]);
        if (diff > EPSILON)
        {
            atomicExch(result, 5);
        }
    }
}

extern "C" __global__ void transpose_naive(const float *A, float *B, int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        B[col * M + row] = A[row * N + col];
    }
}

extern "C" __global__ void matrix_mul(
    const float *A, // Matrix A (M x K)
    const float *B, // Matrix B (K x N)
    float *C,       // Result Matrix C (M x N)
    int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of C that this thread is responsible for
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0;

    // Loop over the tiles of the input matrices
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // 1. Load tiles from Global Memory to Shared Memory
        // Check boundaries for A
        if (row < M && (t * TILE_SIZE + tx) < K)
            ds_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            ds_A[ty][tx] = 0.0;

        // Check boundaries for B
        if (col < N && (t * TILE_SIZE + ty) < K)
            ds_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            ds_B[ty][tx] = 0.0;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // 2. Compute the dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // 3. Write the final result to Global Memory
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

extern "C" __global__ void hadamard_prod(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] * B[idx];
    }
}

extern "C" __global__ void column_reduce(const float *inputMatrix, float *outputSums, int numRows, int numCols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < numCols)
    {
        float sum = 0.0f;

        for (int row = 0; row < numRows; ++row)
        {
            sum += inputMatrix[row * numCols + col];
        }

        outputSums[col] = sum;
    }
}
