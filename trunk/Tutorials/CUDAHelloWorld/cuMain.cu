#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../Common/Rand.h"
#include "../Common/HelloWorld.h"

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

bool MemoryCheck(void * ptr)
{
    if (ptr == NULL)
    {
        fprintf(stdout, "Allocated ptr is a NULL \n");
        return false;
    }
    return true;
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
    float * pC = NULL; // this will be used to read back data from GPU
    float * pD = NULL; // this will be used to store CPU Results

    int nMemoryInBytes = 1024 * 512;
    int nFloatElem = nMemoryInBytes / 4;

    // allocate 4 arrays of 32 Mb each : 
    // on CPU
    pA = (float *) malloc( nMemoryInBytes );
    pB = (float *) malloc( nMemoryInBytes );
    pC = (float *) malloc( nMemoryInBytes );
    pD = (float *) malloc( nMemoryInBytes );

    // on GPU
    cudaMalloc ( (void**) &pCuA, nMemoryInBytes );
    cudaMalloc ( (void**) &pCuB, nMemoryInBytes );
    cudaMalloc ( (void**) &pCuC, nMemoryInBytes );

    if (!MemoryCheck(pCuA) || !MemoryCheck(pCuB) || !MemoryCheck(pCuC))
    {
        fprintf(stdout, "Error: Closing application \n");
        return -1;
    }

    Rand(pA, nFloatElem, 2); // fill array A with random numbers
    Rand(pB, nFloatElem, 3); // fill array B with random numbers

    cudaMemcpy   ( pCuA, pA, nMemoryInBytes, cudaMemcpyHostToDevice );
    cudaMemcpy   ( pCuB, pB, nMemoryInBytes, cudaMemcpyHostToDevice );

    int nThreads[3] = {512, 1, 1};
    int nBlocks[2] = { 64 * 1024 - 1, 1 };
    nBlocks[1] += nFloatElem / (nBlocks[0] * nThreads[0] * nThreads[1] * nThreads[2]);

    float cpuTime = ST_SimpleAddKernel( pA, pB, pD, nFloatElem );
    float gpuTime = CU_SimpleAddKernel( pCuA, pCuB, pCuC, nThreads, nBlocks, nFloatElem );

    // print the cpu and gpu times
    printf("time spent executing by the CPU: %.5f millseconds\n", cpuTime );
    printf("time spent executing by the GPU: %.5f millseconds\n", gpuTime );

    // perform error checking
    cudaMemcpy   ( pC, pCuC, nMemoryInBytes, cudaMemcpyDeviceToHost );

    for (int idx = 0; idx < nFloatElem; idx++)
    {
        if (pC[idx] != pD[idx])
        {
            printf("error %d\n", idx);
            break;
        }
    }

    // free cpu and cuda resources
    free( pA );
    free( pB );
    free( pC );
    free( pD );

    cudaFree( pCuA );
    cudaFree( pCuB );
    cudaFree( pCuC );
	
    return 0;
}