#include "../Common/HelloWorld.h"

__global__ 
void CU_SimpleAddKernel( float * pA, float * pB, float * pC, int n)
{
    int tid = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));

    int bid = (blockIdx.x  + gridDim.x  * (blockIdx.y  + gridDim.z  * blockIdx.z ));
    int blockSize = blockDim.x * blockDim.y * blockDim.z;

    int idx = bid * blockSize + tid;
    
    if (idx < n)
        pC[idx] = pA[idx] + pB[idx];
}


float CU_SimpleAddKernel( float * pA, float * pB, float * pC, int * pthreads, int * pblocks, int n)
{
    dim3 threads = dim3(pthreads[0], pthreads[1], pthreads[2]);
    dim3 blocks  = dim3(pblocks[0], pblocks[1]);

	// create cuda event handles
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
	
	// asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord ( start, 0 );
	
    CU_SimpleAddKernel<<<blocks, threads>>>(pA, pB, pC, n);
	cudaThreadSynchronize();

    cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &gpuTime, start, stop );

    cudaEventDestroy ( start );
    cudaEventDestroy ( stop  );

    return gpuTime;
}
