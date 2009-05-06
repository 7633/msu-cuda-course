#include "Convolution.h"

#include "glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

texture<uchar4, 2, cudaReadModeNormalizedFloat> texRGBA;

uchar4    *g_pOutRGBA = NULL;
cudaArray *g_pInRGBA = NULL;
unsigned   int g_pbo = 0;

static unsigned int g_W = 0;
static unsigned int g_H = 0;

#define SQR(x) ((x) * (x))

__global__ void GaussianBlur(uchar4 * pOut, int w, int h, int r, float sigma)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < w && idy < h)
    {
        // float4 colorSum  = {0.0f, 0.0f, 0.0f, 0.0f}; // uncomment to perform bluring over alpha 
        float4 colorSum  = {0.0f, 0.0f, 0.0f, 1.0f};

        float  weightSum = 0.0f;

        for (float ix = -r; ix <= r; ix++)
            for (float iy = -r; iy <= r; iy++)
            {
                float weight = exp( -(SQR(ix)+SQR(iy)) / SQR(sigma) );
                float4 ic = tex2D( texRGBA, (float) idx + ix, float (idy) + iy );

                colorSum.x += ic.x * weight;
                colorSum.y += ic.y * weight;
                colorSum.z += ic.z * weight;
                // colorSum.w += ic.w * weight; // bluring alpha makes no sense

                weightSum += weight;
            }
        colorSum.x /= weightSum;
        colorSum.y /= weightSum;
        colorSum.z /= weightSum;
        // colorSum.w /= weightSum; // bluring alpha makes no sense

        pOut[ idx + idy * w ] = make_uchar4(colorSum.x*255, 
                                            colorSum.y*255, 
                                            colorSum.z*255, 
                                            colorSum.w*255);
    }
}

extern "C"
{
    bool Wrapper_Convolution_Init(unsigned char * pRGBA, int w, int h, unsigned int pbo)
    {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();

        cudaMalloc( (void **) &g_pOutRGBA, w * h * sizeof(uchar4));
        cudaMallocArray( &g_pInRGBA, &desc, w, h);

        cudaMemcpyToArray( g_pInRGBA, 0, 0, pRGBA, w * h * sizeof(uchar4), cudaMemcpyHostToDevice);

        cudaGLRegisterBufferObject(pbo);

        g_pbo = pbo;

        g_W = w;
        g_H = h;

        cudaError_t error = cudaGetLastError();

        return error == cudaSuccess;
    }

    bool Wrapper_Convolution_Release()
    {
        cudaFree(g_pOutRGBA);

        cudaFreeArray(g_pInRGBA);

        cudaGLUnregisterBufferObject(g_pbo);

        cudaError_t error = cudaGetLastError();

        return error == cudaSuccess;
    }

    bool Wrapper_Convolution_Run(int radius)
    {
        uchar4 * pPBO = NULL;

        cudaGLMapBufferObject( (void **) &pPBO, g_pbo );

        cudaBindTextureToArray(texRGBA, g_pInRGBA);

        dim3 threads(8, 8);
        dim3 blocks((g_W + g_W%threads.x) / threads.x, 
                    (g_H + g_H%threads.y) / threads.y);


        GaussianBlur<<<blocks, threads>>>(pPBO, g_W, g_H, radius, (radius - 1.0f) * 0.5f);
        cudaThreadSynchronize();
        
        cudaGLUnmapBufferObject( g_pbo );

        return true;
    }
};