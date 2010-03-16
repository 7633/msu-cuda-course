#include <stdio.h>
#include <cuda_runtime.h>

#define	PI	3.1415926f

#define SQR(x) ((x)*(x))

#define CUDA_SAFE_CALL(call)                                          \
    {                                                                 \
        cudaResult = call;                                            \
        if (cudaResult != cudaSuccess)                                \
        fprintf(stderr, "cuda error at line %d\n", __LINE__);         \
    }                                                                 \

#define CUDA_CHECK_CALL_SYNC(call)                                    \
    {                                                                 \
        call;                                                         \
        cudaThreadSynchronize();                                      \
        cudaResult = cudaGetLastError();                              \
        if (cudaResult != cudaSuccess)                                \
        fprintf(stderr, "cuda error at line %d\n", __LINE__);         \
    }                                                                 \

#define CUDA_CHECK_CALL_ASYNC(call)                                   \
    {                                                                 \
        call;                                                         \
        cudaResult = cudaGetLastError();                              \
        if (cudaResult != cudaSuccess)                                \
        fprintf(stderr, "cuda error at line %d\n", __LINE__);         \
    }                                                                 \

texture<float, 1, cudaReadModeElementType> g_TexRef;

__global__ void kernel ( float * data )
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data [idx] = tex1Dfetch(g_TexRef, idx);
}

int main ( int argc, char *  argv [] )
{
    cudaError cudaResult = cudaSuccess;

    int nThreads = 64;
    int nBlocks  = 256;
    int nElem    = nThreads * nBlocks;
    int nMemSizeInBytes = nElem * sizeof(float);

    float * phA = NULL;             // host    memory  pointer
    float * phB = NULL;             // host    memory  pointer
    float * pdA = NULL;				// device  memory  pointer
    float * pdB = NULL;				// device  memory  pointer

    phA = (float *) malloc(nMemSizeInBytes);
    phB = (float *) malloc(nMemSizeInBytes);
    CUDA_SAFE_CALL( cudaMalloc ( (void**) &pdA, nMemSizeInBytes ) );
    CUDA_SAFE_CALL( cudaMalloc ( (void**) &pdB, nMemSizeInBytes ) );

    for (int idx = 0; idx < nThreads * nBlocks; idx++)
        phA[idx] = sinf(idx * 2.0f * PI / (nThreads * nBlocks) );

    CUDA_SAFE_CALL( cudaMemcpy ( pdA, phA, nMemSizeInBytes, cudaMemcpyHostToDevice ) );

    CUDA_SAFE_CALL( cudaBindTexture(0, g_TexRef, pdA, nMemSizeInBytes) );

    dim3 threads = dim3( nThreads );
    dim3 blocks  = dim3( nBlocks );

    kernel<<<blocks, threads>>> ( pdB );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    CUDA_SAFE_CALL( cudaMemcpy ( phB, pdB, nMemSizeInBytes, cudaMemcpyDeviceToHost ) );

    for (int idx = 0; idx < nThreads * nBlocks; idx++)
        if (SQR(phA[idx] - phB[idx]) > 0.0001f) printf("a[%d] = %.5f != %.5f = b[%d]\n", idx, phA[idx], phB[idx], idx);

    free(phA);
    free(phB);

    CUDA_SAFE_CALL( cudaFree ( pdA ) );
    CUDA_SAFE_CALL( cudaFree ( pdB ) );

    return 0;
}
