//
// Perform square matrix multiplication using shared memory
//

#include <stdio.h>

#define BLOCK_SIZE 	16			// submatrix size
#define	N			1024		// matrix size is N*N

__global__ void matMult ( float * a, float * b, int n, float * c )
{
	int bx = blockIdx.x;		// block index
	int by = blockIdx.y;

	int tx = threadIdx.x;		// thread index
	int ty = threadIdx.y;
	
								// Index of the first sub-matrix of A processed by the block
	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
								// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;
								// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
								// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * n;
	float sum = 0.0f;			// computed subelement
	
	for ( int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep )
	{
							// Shared memory for the sub-matrix of A
		__shared__ float as [BLOCK_SIZE][BLOCK_SIZE];
							// Shared memory for the sub-matrix of B
		__shared__ float bs [BLOCK_SIZE][BLOCK_SIZE];
		
							// Load the matrices from global memory to shared memory;
		as [ty][tx] = a [ia + n * ty + tx];
		bs [ty][tx] = b [ib + n * ty + tx];
		
		__syncthreads();	// Synchronize to make sure the matrices are loaded
		
							// Multiply the two matrices together;
		for ( int k = 0; k < BLOCK_SIZE; k++ )
			sum += as [ty][k] * bs [k][tx];
			
							// Synchronize to make sure that the preceding
							// computation is done before loading two new
							// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
							// Write the block sub-matrix to global memory;
							// each thread writes one element
	int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	
	c [ic + n * ty + tx] = sum;
}

int main ( int argc, char *  argv [] )
{
	int	numBytes = N * N * sizeof ( float );

					// allocate host memory
    float * a = new float [N*N];
	float * b = new float [N*N];
	float * c = new float [N*N];
	
	for ( int i = 0; i < N; i++ )
		for ( int j = 0; j < N; j++ )
		{
			a [i] = 0.0f;
			b [i] = 1.0f;
		}
		
					// allocate device memory
    float * adev = NULL;
    float * bdev = NULL;
    float * cdev = NULL;
	
    cudaMalloc ( (void**)&adev, numBytes );
    cudaMalloc ( (void**)&bdev, numBytes );
    cudaMalloc ( (void**)&cdev, numBytes );

					// set kernel launch configuration
	dim3 threads ( BLOCK_SIZE, BLOCK_SIZE );
	dim3 blocks  ( N / threads.x, N / threads.y);

					// create cuda event handles
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
	
					// asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord ( start, 0 );
    cudaMemcpy      ( adev, a, numBytes, cudaMemcpyHostToDevice );
    cudaMemcpy      ( bdev, b, numBytes, cudaMemcpyHostToDevice );
	
    matMult<<<blocks, threads>>> ( adev, bdev, N, cdev );
	
    cudaMemcpy      ( c, cdev, numBytes, cudaMemcpyDeviceToHost );
    cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &gpuTime, start, stop );

						// print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime );

					// release resources
    cudaEventDestroy ( start );
    cudaEventDestroy ( stop  );
    cudaFree         ( adev  );
    cudaFree         ( bdev  );
    cudaFree         ( cdev  );

    delete a;
	delete b;
	delete c;

    return 0;
}
