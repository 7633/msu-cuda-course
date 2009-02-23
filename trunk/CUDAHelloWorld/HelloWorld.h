#ifndef _HELLO_WORLD_H_
#define _HELLO_WORLD_H_

float MT_SimpleAddKernel( float * pA, float * pB, float * pC);
float CU_SimpleAddKernel( float * pA, float * pB, float * pC, int * threads, int * blocks, int n);

#endif