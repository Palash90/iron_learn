#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void vector_add(const double *a, const double *b, double *out, int n, int sub)
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

extern "C" __global__ void scaleVector(double *v, double scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        v[idx] *= scale;
    }
}

extern "C" __global__ void updateWeights(double *w, const double *grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        w[idx] -= grad[idx];
    }
}

extern "C" __global__ void accuracyKernel(const double *predictions,
                                          const double *labels,
                                          int *correct,
                                          int n)
{
    __shared__ int block_correct[256]; // adjust to blockDim.x
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int local = 0;

    if (idx < n)
    {
        double prob = predictions[idx];
        double pred_label = (prob >= 0.5 ? 1.0 : 0.0);
        double actual = labels[idx];
        if (fabs(pred_label - actual) < 1e-10)
        {
            local = 1;
        }
    }
    block_correct[tid] = local;
    __syncthreads();

    // Reduce within block
    int stride = blockDim.x / 2;
    while (stride > 0)
    {
        if (tid < stride)
        {
            block_correct[tid] += block_correct[tid + stride];
        }
        __syncthreads();
        stride /= 2;
    }

    if (tid == 0)
    {
        atomicAdd(correct, block_correct[0]);
    }
}

extern "C" __global__ void gemvRowMajor(const double *X, const double *w, double *out,
                                        int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows)
    {
        double sum = 0.0;
        const double *xrow = X + r * cols;
        for (int c = 0; c < cols; ++c)
        {
            sum += xrow[c] * w[c];
        }
        out[r] = sum;
    }
}

extern "C" __global__ void sigmoidKernel(const double *in, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double x = in[i];
        x = fmax(fmin(x, 40.0), -40.0); // clamp
        if (x >= 0.0)
        {
            double z = exp(-x);
            out[i] = 1.0 / (1.0 + z);
        }
        else
        {
            double z = exp(x);
            out[i] = z / (1.0 + z);
        }
    }
}

extern "C" __global__ void thresholdKernel(const double *in, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = (in[i] >= 0.5 ? 1.0 : 0.0);
    }
}

extern "C" __global__ void gradGemvXT(const double *X, const double *loss, double *grad,
                                      int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols)
    {
        double sum = 0.0;
        for (int r = 0; r < rows; ++r)
        {
            sum += X[r * cols + c] * loss[r];
        }
        grad[c] = sum;
    }
}

extern "C" __global__ void compareMemory(const double *a, const double *b, size_t size, int *result)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const double EPSILON = 1e-9;

    if (idx < size)
    {
        double diff = abs(a[idx] - b[idx]);
        if (diff > EPSILON)
        {
            atomicExch(result, 0);
        }
    }
}

extern "C" __global__ void transpose_naive(const float *A, float *B, int M, int N)
{
    // 1. Calculate the coordinates (row, col) for the output matrix B
    int col_b = blockIdx.x * blockDim.x + threadIdx.x; // Column index in B
    int row_b = blockIdx.y * blockDim.y + threadIdx.y; // Row index in B

    // Check if the thread is within the bounds of the output matrix B (N x M)
    if (row_b < N && col_b < M)
    {
        // 2. Transposition Rule: B[row_b][col_b] = A[col_b][row_b]

        // Input Matrix A (M x N) index: row_a = col_b, col_a = row_b
        int index_a = col_b * N + row_b;

        // Output Matrix B (N x M) index:
        int index_b = row_b * M + col_b;

        // Perform the transpose
        B[index_b] = A[index_a];
    }
}

extern "C" __global__ void matrixMul(
    const double *A,  // Matrix A (M x K)
    const double *B,  // Matrix B (K x N)
    double *C,        // Result Matrix C (M x N)
    int M,            // Height of C (rows of A)
    int N,            // Width of C (columns of B)
    int K             // Inner dimension
)
{
    // Map thread indices to the row and column of the output matrix C.
    // row (i) is calculated along the Y dimension (BlockIdx.y, ThreadIdx.y)
    // col (j) is calculated along the X dimension (BlockIdx.x, ThreadIdx.x)
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds: Ensure the thread is within the dimensions of the result matrix C (M x N)
    if (row < M && col < N) {
        
        double sum = 0.0;
        
        // Loop over the inner dimension K to compute the dot product
        // C[row][col] = sum over k ( A[row][k] * B[k][col] )
        for (int k = 0; k < K; ++k) {
            
            // A[row][k] is at index (row * K + k)
            double a_val = A[row * K + k];
            
            // B[k][col] is at index (k * N + col)
            double b_val = B[k * N + col];
            
            sum += a_val * b_val;
        }

        // Write the result to the output matrix C[row][col]
        C[row * N + col] = sum;
    }
}