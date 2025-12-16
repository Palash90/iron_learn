#include <cuda.h>
#include <cuda_runtime.h>
#define TILE_SIZE 16

extern "C" __global__ void fill_value(double *out, int n, double value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = value;
    }
}

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

extern "C" __global__ void element_op(const double *s, double *r, int n, int op, double scale)
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
            r[idx] = exp(s[idx]) / (1 + exp(s[idx]));
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

extern "C" __global__ void transpose_naive(const double *A, double *B, int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        B[col * M + row] = A[row * N + col];
    }
}

extern "C" __global__ void matrixMulTiled(
    const double *A, // Matrix A (M x K)
    const double *B, // Matrix B (K x N)
    double *C,       // Result Matrix C (M x N)
    int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ double ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ double ds_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of C that this thread is responsible for
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

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

extern "C" __global__ void hadamardProd(const double *A, const double *B, double *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] * B[idx];
    }
}

extern "C" __global__ void reluKernel(const double *in, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Equivalent to out[i] = (in[i] > 0.0) ? in[i] : 0.0;
        out[i] = fmax(0.0, in[i]);
    }
}

extern "C" __global__ void reluDerivativeKernel(const double *in, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // out[i] = (in[i] > 0.0) ? 1.0 : 0.0;
        out[i] = (in[i] > 0.0) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void sigmoidPrimeKernel(const double *S, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // S is the sigmoid output S(x)
        out[i] = S[i] * (1.0 - S[i]);
    }
}

extern "C" __global__ void sumReductionKernel(const double *in, double *out, int n)
{
    // Uses a simple shared memory reduction within each block
    __shared__ double sdata[1024]; // Assuming max blockDim.x is 1024

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? in[idx] : 0.0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result of block to global memory (out)
    if (tid == 0)
    {
        // Note: 'out' must be sized to have one element per block (gridDim.x)
        out[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void logLossDerivativeKernel(
    const double *predicted,
    const double *actual,
    double *out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double p = predicted[i];
        double a = actual[i];

        // Use a constant for epsilon
        const double EPSILON = 1e-15;

        // Clip predicted values for stability
        if (p < EPSILON)
            p = EPSILON;
        if (p > (1.0 - EPSILON))
            p = (1.0 - EPSILON);

        // Calculate derivative: -(a / p) + (1.0 - a) / (1.0 - p)
        out[i] = -(a / p) + (1.0 - a) / (1.0 - p);
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
