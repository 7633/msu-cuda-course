#include <cuda.h>
#include <stdio.h>

int main(int argc, char ** argv)
{
	int dev_count = 0;

	CUdevice   device;
	CUcontext  context;
	CUmodule   module;
	CUfunction function;

	cuInit(0);

	cuDeviceGetCount(&dev_count);

	if (dev_count < 1) return -1;

	cuDeviceGet( &device, 0 );
	cuCtxCreate( &context, 0, device );
	
	cuModuleLoad( &module, "hello.cuda_runtime.ptx" );
	cuModuleGetFunction( &function, module, "_Z6kernelPf" );

	int N = 512;
	CUdeviceptr pData;
	cuMemAlloc( &pData, N * sizeof(float) );
	cuFuncSetBlockShape( function, N, 1, 1 );
	cuParamSeti( function, 0, pData );
	cuParamSetSize( function, 4 );

	cuLaunchGrid( function, 1, 1 );

	float * pHostData = new float[N];

	cuMemcpyDtoH( pHostData, pData, N * sizeof( float) );

	cuMemFree( pData );

	delete [] pHostData;

	return 0;
}