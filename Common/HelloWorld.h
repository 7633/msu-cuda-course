#ifndef _HELLO_WORLD_H_
#define _HELLO_WORLD_H_

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
void  MT_SimpleAddKernel2(void * pVoid);
float ST_SimpleAddKernel( float * pA, float * pB, float * pC, int n);
float CU_SimpleAddKernel( float * pA, float * pB, float * pC, int * threads, int * blocks, int n);

#endif