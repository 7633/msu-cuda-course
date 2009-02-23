#include <stdio.h>

#define	N	(256*256)		// array size
#define	PI	3.1415926f

__global__ void kernel ( float * data )
{ 
   int 		idx = blockIdx.x * blockDim.x + threadIdx.x;
   float	x   = 2.0f * PI * (float) idx / (float) N;
   
   data [idx] = sin ( sqrtf ( x ) );
}

int main ( int argc, char *  argv [] )
{
    float * a   = new float [N];	// CPU memory
    float * dev = NULL;				// GPU memory
		
									// allocate device memory
    cudaMalloc ( (void**)&dev, N * sizeof ( float ) );

    dim3 threads = dim3( 512, 1 );
    dim3 blocks  = dim3( N / threads.x, 1 );
					
    kernel<<<blocks, threads>>> ( dev );
    cudaThreadSynchronize();

    cudaMemcpy ( a, dev, N * sizeof ( float ), cudaMemcpyDeviceToHost );
    cudaFree   ( dev   );

    for (int idx = 0; idx < N; idx++)
        printf("a[%d] = %.5f\n", idx, a[idx]);

    delete a;

    return 0;
}
