#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <process.h> 
#include <xmmintrin.h>

#include "../Common/HelloWorld.h"
#include "../Common/Rand.h"

int main ( int argc, char *  argv [] )
{
    float * pA = NULL;
    float * pB = NULL;
    float * pC = NULL; 
    float * pD = NULL; 

    int nMemoryInBytes = 1024 * 1024;
    int nFloatElem = nMemoryInBytes / 4;

    // allocate 4 arrays of 512 Kb each : 
    pA = (float *) malloc( nMemoryInBytes );
    pB = (float *) malloc( nMemoryInBytes );
    pC = (float *) malloc( nMemoryInBytes );
    pD = (float *) malloc( nMemoryInBytes );

    __m128 * pSseA = (__m128 *) pA;
    __m128 * pSseB = (__m128 *) pB;
    __m128 * pSseC = (__m128 *) pD; 

    Rand(pA, nFloatElem, 2); // fill array A with random numbers
    Rand(pB, nFloatElem, 3); // fill array B with random numbers
    
    float cpuTime = ST_SimpleAddKernel( pA, pB, pC, nFloatElem, 1000 );
    float sseTime = SSE_SimpleAddKernel( pSseA, pSseB, pSseC, nFloatElem / 4, 1000);

    // print the cpu and ssse times
    printf("time spent executing by the CPU: %.5f millseconds\n", cpuTime );
    printf("time spent executing by the SSE: %.5f millseconds\n", sseTime );

    // perform error checking    
    for (int idx = 0; idx < nFloatElem; idx++)
    {
        if (pC[idx] != pD[idx])
        {
            printf("error %d\n", idx);
            break;
        }
    }

    // free cpu resources
    free( pA );
    free( pB );
    free( pC );
    free( pD );

    return 0;

}