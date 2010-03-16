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

texture<float, 2, cudaReadModeElementType> g_TexRef;

__global__ void kernel ( float * data )
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data [idx + blockIdx.y * gridDim.x * blockDim.x] = tex2D(g_TexRef, idx, blockIdx.y);
}

int main ( int argc, char *  argv [] )
{
    cudaError cudaResult = cudaSuccess;

    int nThreads  = 64;
    int nBlocksX  = 256;
    int nBlocksY  = 2;
    int nElem    = nThreads * nBlocksX * nBlocksY;
    int nMemSizeInBytes = nElem * sizeof(float);

    float * phA = NULL;             // host    memory    pointer
    float * phB = NULL;             // host    memory    pointer
    float * pdA = NULL;				// device  memory    pointer
    float * pdB = NULL;				// device  memory    pointer
    cudaArray * paA = NULL;         // device  cudaArray pointer

    cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    phA = (float *) malloc(nMemSizeInBytes);
    phB = (float *) malloc(nMemSizeInBytes);
    CUDA_SAFE_CALL( cudaMalloc ( (void**) &pdA, nMemSizeInBytes ) );
    CUDA_SAFE_CALL( cudaMalloc ( (void**) &pdB, nMemSizeInBytes ) );
    CUDA_SAFE_CALL( cudaMallocArray(&paA, &cfDesc, nBlocksX * nThreads, nBlocksY) );

    for (int idx = 0; idx < nThreads * nBlocksX; idx++)
        phA[idx] = sinf(idx * 2.0f * PI / (nThreads * nBlocksX) );
    
    for (int idx = 0; idx < nThreads * nBlocksX; idx++)
        phA[idx + nThreads * nBlocksX] = cosf(idx * 2.0f * PI / (nThreads * nBlocksX) );

    CUDA_SAFE_CALL( cudaMemcpyToArray ( paA, 0, 0, phA, nMemSizeInBytes, cudaMemcpyHostToDevice ) );

    CUDA_SAFE_CALL( cudaBindTextureToArray(g_TexRef, paA) );

    dim3 threads = dim3( nThreads );
    dim3 blocks  = dim3( nBlocksX, nBlocksY );

    kernel<<<blocks, threads>>> ( pdB );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    CUDA_SAFE_CALL( cudaMemcpy ( phB, pdB, nMemSizeInBytes, cudaMemcpyDeviceToHost ) );

    for (int idx = 0; idx < nThreads * nBlocksX * nBlocksY; idx++)
        if (SQR(phA[idx] - phB[idx]) > 0.0001f) printf("a[%d] = %.5f != %.5f = b[%d]\n", idx, phA[idx], phB[idx], idx);

    free(phA);
    free(phB);

    CUDA_SAFE_CALL( cudaFree ( pdA ) );
    CUDA_SAFE_CALL( cudaFree ( pdB ) );
    CUDA_SAFE_CALL( cudaFreeArray( paA ) );

    return 0;
}
