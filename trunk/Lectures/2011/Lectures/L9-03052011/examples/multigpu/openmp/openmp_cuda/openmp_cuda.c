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
 * GPUs and CPU using OpenMP.
 *
 */

#include "pattern2d.h"

#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The thread configuration structure.
typedef struct
{
	int idevice;
	int status;
	int step;
	int nx, ny;
	float *inout_cpu, *in_dev, *out_dev;
}
config_t;

int nticks = 10;

// The size of memory region.
int nx = 128, ny = 128;

// The function executed by each thread assigned with CUDA device.
int thread_func(config_t* config)
{
	int idevice = config->idevice;
	
	// Set focus on device with the specificed index.
	// (Will implcitly create CUDA context and make
	// it current).
	cudaError_t cuda_status = cudaSetDevice(idevice);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot set focus to device %d, status = %d\n",
			cuda_status);
		return cuda_status;
	}
	
	size_t size = config->nx * config->ny * sizeof(float);

	// Create device arrays for input and output data.
	cuda_status = cudaMalloc((void**)&config->in_dev, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate CUDA input buffer on device %d, status = %d\n",
			idevice, cuda_status);
		return cuda_status;
	}
	cuda_status = cudaMalloc((void**)&config->out_dev, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate CUDA output buffer on device %d, status = %d\n",
			idevice, cuda_status);
		return cuda_status;
	}
	
	// Copy input data to device buffer.
	cuda_status = cudaMemcpy(config->in_dev, config->inout_cpu, size,
		cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy input data to CUDA buffer on device %d, status = %d\n",
			idevice, cuda_status);
		return cuda_status;
	}
       
        printf("Device %d initialized\n", idevice);

	// Compute the given number of steps.
	for (int i = 0; i < nticks; i++)
	{
		int status = pattern2d_gpu(1, config->nx, 1, 1, config->ny, 1,
			config->in_dev, config->out_dev, idevice);
		if (status)
		{
			fprintf(stderr, "Cannot execute pattern 2d on device %d, status = %d\n",
				idevice, status);
			return status;
		}
		config->step++;
	
		// Swap device input and output buffers.
		float* swap = config->in_dev;
		config->in_dev = config->out_dev;
		config->out_dev = swap;

		printf("Device %d completed step %d\n", idevice, config->step);
	}

	// Offload results back to shared memory.
	cuda_status = cudaMemcpy(config->inout_cpu, config->in_dev, size,
		cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy output data from CUDA buffer on device %d, status = %d\n",
			idevice, cuda_status);
		return cuda_status;
	}

	// Dispose device buffers.
	cuda_status = cudaFree(config->in_dev);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot release input buffer on device %d, status = %d\n",
			idevice, cuda_status);
		return cuda_status;
	}
	cuda_status = cudaFree(config->out_dev);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot release output buffer on device %d, status = %d\n",
			idevice, cuda_status);
		return cuda_status;
	}

        printf("Device %d deinitialized\n", idevice);

	return 0;
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

	// Create input data. Let each device to have an equal piece
	// of single shared data array.
	size_t np = nx * ny;
	float* inout = (float*)malloc((ndevices + 2) * np * sizeof(float));
	float invdrandmax = 1.0 / RAND_MAX;
	for (size_t i = 0; i < np; i++)
		inout[i] = rand() * invdrandmax;
	for (int i = 0; i < ndevices + 1; i++)
		memcpy(inout + np * (i + 1), inout, np * sizeof(float));

	// Create workers configs. Its data will be passed as
	// argument to thread_func.
	config_t* configs = (config_t*)malloc(
		sizeof(config_t) * ndevices);
	
	// For each CUDA device found create a separate thread
	// and execute the thread_func.
	float* control;
	#pragma omp sections
	{
		// Section for GPU threads.
		#pragma omp section
		{
			#pragma omp parallel for
			for (int i = 0; i < ndevices; i++)
			{
				config_t* config = configs + i;
				config->idevice = i;
				config->step = 0;
				config->nx = nx; config->ny = ny;
				config->inout_cpu = inout + np * i;	
				config->status = thread_func(config);
			}
		}
		
		// Section for CPU thread.
		#pragma omp section
		{	
			// In parallel main thread launch CPU function equivalent
			// to CUDA kernels, to check the results.
			control = inout + ndevices * np;
			float* input = inout + (ndevices + 1) * np;
			for (int i = 0; i < nticks; i++)
			{
				pattern2d_cpu(1, configs->nx, 1, 1, configs->ny, 1,
					input, control, ndevices);
				float* swap = control;
				control = input;
				input = swap;
			}
			float* swap = control;
			control = input;
			input = swap;
		}
	}

	for (int i = 0; i < ndevices; i++)
	{
		int status = configs[i].status;
		if (status)
		{
			fprintf(stderr, "Cannot execute thread function for device %d, status = %d\n",
				i, status);
			return status;
		}
	}
	
	// Compare each GPU result to CPU result.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		// Find the maximum abs difference.
		int maxi = 0, maxj = 0;
		float maxdiff = fabs(control[0] - (inout + idevice * np)[0]);
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				float diff = fabs(
					control[i + j * nx] -
					(inout + idevice * np)[i + j * nx]);
				if (diff > maxdiff)
				{
					maxdiff = diff;
					maxi = i; maxj = j;
				}
			}
		}
		printf("Device %d result abs max diff = %f @ (%d,%d)\n",
			idevice, maxdiff, maxi, maxj);
	}
	
	free(configs);
	free(inout);
	
	return 0;
}

