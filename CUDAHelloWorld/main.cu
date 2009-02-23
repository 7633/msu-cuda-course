#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "HelloWorld.cu"

int GetDeviceInfo(int & deviceCount, cudaDeviceProp & deviceProp, bool bPrintProp = false);

int GetDeviceInfo(int & deviceCount, cudaDeviceProp & deviceProp, bool bPrintProp)
{
	cudaGetDeviceCount ( &deviceCount );
	
    if (bPrintProp)
    {
	    fprintf(stdout, "Found %d devices\n", deviceCount );
    	
	    for ( int device = 0; device < deviceCount; device++ )
	    {
		    cudaGetDeviceProperties ( &deviceProp, device );
    		
		    fprintf(stdout, "Device %d\n", device );
		    fprintf(stdout, "Compute capability     : %d.%d\n", deviceProp.major, deviceProp.minor );
		    fprintf(stdout, "Name                   : %s   \n", deviceProp.name );
		    fprintf(stdout, "Total Global Memory    : %d   \n", deviceProp.totalGlobalMem );
		    fprintf(stdout, "Shared memory per block: %d   \n", deviceProp.sharedMemPerBlock );
		    fprintf(stdout, "Registers per block    : %d   \n", deviceProp.regsPerBlock );
		    fprintf(stdout, "Warp size              : %d   \n", deviceProp.warpSize );
		    fprintf(stdout, "Max threads per block  : %d   \n", deviceProp.maxThreadsPerBlock );
		    fprintf(stdout, "Total constant memory  : %d   \n", deviceProp.totalConstMem );
	    }
    }

    return deviceCount;
}

void Rand(float * pArray, int n, int x)
{
    srand(108719 * x + 1);

    for (int ip = 0; ip < n; ip++)
    {
        pArray[ip] = rand() % 1025 - 512;
    }
}

int main ( int argc, char *  argv [] )
{
	int deviceCount;
    cudaDeviceProp deviceProp;
    
    GetDeviceInfo(deviceCount, deviceProp, true);

    float * pCuA = NULL;
    float * pCuB = NULL;
    float * pCuC = NULL;

    float * pA = NULL;
    float * pB = NULL;
    float * pC = NULL;

    int nMemoryInBytes = 1024 * 1024 * 16;
    int nFloatElem = nMemoryInBytes / 4;

    // allocate 3 arrays of 16 Mb  each : 
    // on CPU
    pA = (float *) malloc( nMemoryInBytes );
    pB = (float *) malloc( nMemoryInBytes );
    pC = (float *) malloc( nMemoryInBytes );

    // on GPU
    cudaMalloc ( (void**) &pCuA, nMemoryInBytes );
    cudaMalloc ( (void**) &pCuB, nMemoryInBytes );
    cudaMalloc ( (void**) &pCuC, nMemoryInBytes );

    Rand(pA, nFloatElem, 2);
    Rand(pB, nFloatElem, 3);

    cudaMemcpy   ( pCuA, pA, nMemoryInBytes, cudaMemcpyHostToDevice );
    cudaMemcpy   ( pCuB, pB, nMemoryInBytes, cudaMemcpyHostToDevice );
    //cudaMemcpy      ( pCuC, pC, nMemoryInBytes, cudaMemcpyHostToDevice );

    int nThreads[3] = {256, 1, 1};
    int nBlocks[2] = { 32 * 1024, 1 };
    nBlocks[1] += nFloatElem / (nBlocks[0] * nThreads[0] * nThreads[1] * nThreads[2]);
    
    float gpuTime = CU_SimpleAddKernel( pCuA, pCuB, pCuC, nThreads, nBlocks, nFloatElem);

	// print the cpu and gpu times
    printf("time spent executing by the GPU: %.5f millseconds\n", gpuTime );


    //cudaMemcpy   ( pA, pCuA, nMemoryInBytes, cudaMemcpyDeviceToHost );
    //cudaMemcpy   ( pB, pCuB, nMemoryInBytes, cudaMemcpyDeviceToHost );
    cudaMemcpy   ( pC, pCuC, nMemoryInBytes, cudaMemcpyDeviceToHost );

    for (int ip = 0; ip < nFloatElem; ip++)
    {
        if (pC[ip] != pA[ip] + pB[ip])
            printf("error %d\n", ip);
    }

    // free cpu and cuda resources
    free( pA );
    free( pB );
    free( pC );

    cudaFree( pCuA );
    cudaFree( pCuB );
    cudaFree( pCuC );
	
    return 0;
}