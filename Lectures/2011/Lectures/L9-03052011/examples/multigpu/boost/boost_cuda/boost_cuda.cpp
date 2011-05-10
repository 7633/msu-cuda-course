/*
 * MSU CUDA Course Examples and Exercises.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * without any restrictons.
 *
 * This sample demonstates parallel execution of CUDA programs on multiple
 * GPUs and CPU using Boost. GPUs are synchronized after each step.
 *
 */

#include "boost_cuda.h"
#include "pattern2d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nticks = 10;

// The size of memory region.
int nx = 128, ny = 128;

int ThreadRunner::status = 0;
boost::mutex ThreadRunner::m;

void ThreadRunner::SetLastError(int status)
{
	boost::mutex::scoped_lock lock(m);
	if (status)
		ThreadRunner::status = status;
}
	
int ThreadRunner::GetLastError()
{
	boost::mutex::scoped_lock lock(m);
	int status = ThreadRunner::status;
	ThreadRunner::status = 0;
	return status;
}

// The function executed by each thread assigned with CUDA device.
void ThreadRunner::thread_func()
{
	// Thread iterations loop.
	while (1)
	{
		this->b2.wait();

		// Destructor signals "finish" to end thread loop.
		if (finish) break;

		int idevice = this->idevice;

		// Set focus on the specified CUDA context.
		// Previously we created one context for each thread.
		CUresult cu_status = cuCtxPushCurrent(ctx);
		if (cu_status != CUDA_SUCCESS)
		{
			fprintf(stderr, "Cannot push current context for device %d, status = %d\n",
				idevice, cu_status);
			ThreadRunner::SetLastError((int)cu_status);
		}

		int status = pattern2d_gpu(1, nx, 1, 1, ny, 1, in_dev, out_dev, idevice);
		if (status)
		{
			fprintf(stderr, "Cannot execute pattern 2d on device %d, status = %d\n",
				idevice, status);
			ThreadRunner::SetLastError(status);
		}
		step++;

		// Pop the previously pushed CUDA context out of this thread.
		cu_status = cuCtxPopCurrent(&ctx);
		if (cu_status != CUDA_SUCCESS)
		{
			fprintf(stderr, "Cannot pop current context for device %d, status = %d\n",
				idevice, cu_status);
			ThreadRunner::SetLastError((int)cu_status);
		}

		// Swap device input and output buffers.
		float* swap = in_dev;
		in_dev = out_dev;
		out_dev = swap;

		printf("Device %d completed step %d\n", idevice, step);
	
		this->b1->wait();
	}
}

ThreadRunner::ThreadRunner(int idevice, int nx, int ny, boost::barrier* b) :
	t(boost::bind(&ThreadRunner::thread_func, this)), b2(2), finish(0)
{
	this->idevice = idevice;
	this->step = 0;
	this->nx = nx; this->ny = ny;
	this->b1 = b;
	
	CUdevice dev;		
	CUresult cu_status = cuDeviceGet(&dev, idevice);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot get CUDA device by index %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}
	
	cu_status = cuCtxCreate(&ctx, 0, dev);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot create a context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}

	// Create device arrays for input and output data.
	size_t size = nx * ny * sizeof(float);
	cudaError_t cuda_status = cudaMalloc((void**)&in_dev, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate CUDA input buffer on device %d, status = %d\n",
			idevice, cuda_status);
		ThreadRunner::SetLastError((int)cuda_status);
	}
	cuda_status = cudaMalloc((void**)&out_dev, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate CUDA output buffer on device %d, status = %d\n",
			idevice, cuda_status);
		ThreadRunner::SetLastError((int)cuda_status);
	}

	// Pop the previously pushed CUDA context out of this thread.
	cu_status = cuCtxPopCurrent(&ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot pop current context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}
      
	printf("Device %d initialized\n", idevice);
}

// Load input data on device.
void ThreadRunner::Load(float* input)
{
	// Set focus on the specified CUDA context.
	// Previously we created one context for each thread.
	CUresult cu_status = cuCtxPushCurrent(ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot push current context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}

	// Copy input data to device buffer.
	size_t size = nx * ny * sizeof(float);
	cudaError_t cuda_status = cudaMemcpy(in_dev, input, size,
		cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy input data to CUDA buffer on device %d, status = %d\n",
			idevice, cuda_status);
		ThreadRunner::SetLastError((int)cuda_status);
	}

	// Pop the previously pushed CUDA context out of this thread.
	cu_status = cuCtxPopCurrent(&ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot pop current context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}
}

// Unload output data from device.
void ThreadRunner::Unload(float* output)
{
	// Set focus on the specified CUDA context.
	// Previously we created one context for each thread.
	CUresult cu_status = cuCtxPushCurrent(ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot push current context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}

	// Offload results back to host memory.
	size_t size = nx * ny * sizeof(float);
	cudaError_t cuda_status = cudaMemcpy(output, in_dev, size,
		cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy output data from CUDA buffer on device %d, status = %d\n",
			idevice, cuda_status);
		ThreadRunner::SetLastError((int)cuda_status);
	}

	// Pop the previously pushed CUDA context out of this thread.
	cu_status = cuCtxPopCurrent(&ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot pop current context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}
}

// Dispose device buffers and destroy used contexts.
ThreadRunner::~ThreadRunner()
{
	// Signal thread_func to finish.
	this->finish = 1;
	this->b2.wait();

	// Set focus on the specified CUDA context.
	// Previously we created one context for each thread.
	CUresult cu_status = cuCtxPushCurrent(ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot push current context for device %d, status = %d\n",
			idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}

	cudaError_t cuda_status = cudaFree(in_dev);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot release input buffer on device %d, status = %d\n",
			idevice, cuda_status);
		ThreadRunner::SetLastError((int)cuda_status);
	}
	cuda_status = cudaFree(out_dev);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot release output buffer on device %d, status = %d\n",
			idevice, cuda_status);
		ThreadRunner::SetLastError((int)cuda_status);
	}

	cu_status = cuCtxDestroy(ctx);
	if (cu_status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot destroy context for device %d\n", idevice, cu_status);
		ThreadRunner::SetLastError((int)cu_status);
	}

	printf("Device %d deinitialized\n", idevice);
}

void ThreadRunner::Pass()
{
	this->b2.wait();
}

int main(int argc, char* argv[])
{
	int ndevices = 0;
	cudaError_t cuda_status = cudaGetDeviceCount(&ndevices);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot get the cuda device count, status = %d\n",
			cuda_status);
		return cuda_status;
	}
	
	// Return if no cuda devices present.
	printf("%d CUDA device(s) found\n", ndevices);
	if (!ndevices) return 0;
	
	// Create input data. Each device will have an equal
	// piece of single shared data array.
	size_t np = nx * ny;
	float* data = new float[np * 2];
	float invdrandmax = 1.0 / RAND_MAX;
	for (size_t i = 0; i < np; i++)
		data[i] = rand() * invdrandmax;

	// Create a barrier that will wait for (ndevices + 1)
	// invocations of wait().
	boost::barrier b(ndevices + 1);
	
	// Initialize thread runners and load input data.
	ThreadRunner** runners = new ThreadRunner*[ndevices + 1];
	for (int i = 0; i < ndevices; i++)
	{
		runners[i] = new ThreadRunner(i, nx, ny, &b);
		runners[i]->Load(data);
	}

	// Compute the given number of steps.
	float* input = data;
	float* output = data + np;
	for (int i = 0; i < nticks; i++)
	{
		// Pass iteration on device threads.
		for (int i = 0; i < ndevices; i++)
			runners[i]->Pass();
		
		int status = ThreadRunner::GetLastError();
		if (status) return status;

		// In parallel main thread launch CPU function equivalent
		// to CUDA kernels, to check the results.
		pattern2d_cpu(1, nx, 1, 1, ny, 1,
			input, output, ndevices);
		float* swap = output;
		output = input;
		input = swap;

		b.wait();		
	}

	// Compare each GPU result to CPU result.
	float* control = input;
	float* result = output;
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		runners[idevice]->Unload(result);

		// Find the maximum abs difference.
		int maxi = 0, maxj = 0;
		float maxdiff = fabs(control[0] - result[0]);
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				float diff = fabs(
					control[i + j * nx] - result[i + j * nx]);
				if (diff > maxdiff)
				{
					maxdiff = diff;
					maxi = i; maxj = j;
				}
			}
		}
		printf("Device %d result abs max diff = %f @ (%d,%d)\n",
			idevice, maxdiff, maxi, maxj);
		
		delete runners[idevice];
	}
	
	delete[] data;
	delete[] runners;
	
	return 0;
}

