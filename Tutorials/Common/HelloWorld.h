#ifndef _HELLO_WORLD_H_
#define _HELLO_WORLD_H_


#include <xmmintrin.h>

struct SArgList
{
    float * pA;
    float * pB;
    float * pC;
    int start; // 
    int stop;  //
    int times;
};

void  MT_SimpleAddKernel( void * pVoid);
float SSE_SimpleAddKernel( __m128 * pA, __m128 * pB, __m128 * pC, int n, int times = 100);
float ST_SimpleAddKernel( float * pA, float * pB, float * pC, int n, int times = 100);
float CU_SimpleAddKernel( float * pA, float * pB, float * pC, int * threads, int * blocks, int n, int times = 100);

#endif