#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__

void matrix_mul(const float* A, const float* B, float* C, int N) {
    // To DO: Device a row major indexing
	int rowID = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int elemID;											// Element address

    // a_ij = a[i][j], where a is in row major order
	if(rowID < N && colID < N){
		elemID = colID + rowID * N; 				
		C[elemID] = A[elemID] + B[elemID];
	}
}



