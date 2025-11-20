#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix size (N x N)
#define N 40

// CUDA kernel for matrix multiplication
extern "C" __global__ void matrixMulKernel(double *A, double *B, double *C, int numARows, int numAColumns, int numBColumns)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < numARows && Col < numBColumns)
    {
        float Cvalue = 0.0;

        for (int k = 0; k < numAColumns; ++k)
        {
            Cvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        }

        C[Row * numBColumns + Col] = Cvalue;
    }
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void printMatrix(double *A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Access element at row i, column j
            printf("%8.3f ", A[i * cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    int size = N * N * sizeof(float);

    // Host matrices
    double h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = (float)(i + 1); // 1, 2, 3...
        h_B[i] = (float)(i + 1);
    }

    printf("Matrix A:\n");
    printMatrix(h_A, N, N);
    printf("Matrix B:\n");
    printMatrix(h_B, N, N);

    // Device matrices
    double *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void **)&d_A, size), "Allocating d_A");
    checkCudaError(cudaMalloc((void **)&d_B, size), "Allocating d_B");
    checkCudaError(cudaMalloc((void **)&d_C, size), "Allocating d_C");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Copying A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Copying B to device");

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, N, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Copying C to host");

    // Print result
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%6.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
