#ifndef _HELLO_WORLD_H_
#define _HELLO_WORLD_H_

float ST_SimpleAddKernel( float * pA, float * pB, float * pC, int n);
float CU_SimpleAddKernel( float * pA, float * pB, float * pC, int * threads, int * blocks, int n);

#endif