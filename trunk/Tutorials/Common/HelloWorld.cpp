#include <time.h>

#include "HelloWorld.h"

float ST_SimpleAddKernel( float * pA, float * pB, float * pC, int n, int times)
{
    clock_t start, stop;

    start = clock();

    for (int itimes = 0; itimes < times; itimes++)
    {
        for (int idx = 0; idx < n; idx++)
        {
            pC[idx] = pA[idx] + pB[idx];
        }
    }

    stop = clock();

    float cpuTime = (stop - start) * 1000.0f / CLOCKS_PER_SEC;

    return cpuTime;
}

void MT_SimpleAddKernel(void * pVoid)
{
    SArgList *pArgList = (SArgList *) pVoid;

    float * pA = pArgList->pA;
    float * pB = pArgList->pB;
    float * pC = pArgList->pC;
    int start = pArgList->start;
    int stop = pArgList->stop;
    int times = pArgList->times; 

    for (int itimes = 0; itimes < times; itimes++)
    {
        for (int idx = start ; idx < stop; idx++)
        {
            pC[idx] = pA[idx] + pB[idx];
        }
    }
}

float SSE_SimpleAddKernel( __m128 * pA, __m128 * pB, __m128 * pC, int n, int times)
{
    clock_t start, stop;

    start = clock();    

    for (int itimes = 0; itimes < times; itimes++)
    {
        for (int idx = 0; idx < n; idx++)
        {
            pC[idx] = _mm_add_ps( pA[idx], pB[idx] );        
        }
    }

    stop = clock();

    float sseTime = (stop - start) * 1000.0f / CLOCKS_PER_SEC;

    return sseTime;
}
