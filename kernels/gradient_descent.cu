#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
extern "C" __global__ void matrixMulKernel(double *A, double *B, double *C, int numARows, int numAColumns, int numBColumns)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < numARows && Col < numBColumns)
    {
        double Cvalue = 0.0;

        for (int k = 0; k < numAColumns; ++k)
        {
            Cvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        }

        C[Row * numBColumns + Col] = Cvalue;
    }
}

extern "C" __global__
void vectorSub(const double* a, const double* b, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__
void scaleVector(double* v, double scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] *= scale;
    }
}

extern "C" __global__
void updateWeights(double* w, const double* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        w[idx] -= grad[idx];
    }
}

extern "C" __global__
void accuracyKernel(const double* predictions,
                    const double* labels,
                    int* correct,
                    int n) {
    __shared__ int block_correct[256]; // adjust to blockDim.x
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int local = 0;

    if (idx < n) {
        double prob = predictions[idx];
        double pred_label = (prob >= 0.5 ? 1.0 : 0.0);
        double actual = labels[idx];
        if (fabs(pred_label - actual) < 1e-10) {
            local = 1;
        }
    }
    block_correct[tid] = local;
    __syncthreads();

    // Reduce within block
    int stride = blockDim.x / 2;
    while (stride > 0) {
        if (tid < stride) {
            block_correct[tid] += block_correct[tid + stride];
        }
        __syncthreads();
        stride /= 2;
    }

    if (tid == 0) {
        atomicAdd(correct, block_correct[0]);
    }
}

extern "C" __global__
void gemvRowMajor(const double* X, const double* w, double* out,
                  int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        double sum = 0.0;
        const double* xrow = X + r * cols;
        for (int c = 0; c < cols; ++c) {
            sum += xrow[c] * w[c];
        }
        out[r] = sum;
    }
}

extern "C" __global__
void sigmoidKernel(const double* in, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        x = fmax(fmin(x, 40.0), -40.0); // clamp
        if (x >= 0.0) {
            double z = exp(-x);
            out[i] = 1.0 / (1.0 + z);
        } else {
            double z = exp(x);
            out[i] = z / (1.0 + z);
        }
    }
}

extern "C" __global__
void thresholdKernel(const double* in, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (in[i] >= 0.5 ? 1.0 : 0.0);
    }
}