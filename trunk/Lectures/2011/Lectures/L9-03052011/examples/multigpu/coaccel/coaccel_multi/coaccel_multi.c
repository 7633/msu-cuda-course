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
 * GPUs and CPU using COACCEL multi API.
 *
 */

#include "pattern2d.h"

#include <coaccel.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The thread configuration structure.
typedef struct
{
	int step;
	int nx, ny;
	float *inout_cpu, *in_dev, *out_dev;
}
config_t;

int nticks = 10;

// The size of memory region.
int nx = 128, ny = 128;

// Allocate space for each device.
// This routine is executed only once by coaccel_multi_init() call.
// All device memory allocated here will be resident until
// execution of deinit() by coaccel_multi_finalize() call.
int init(coaccel_device_group group, coaccel_device device,
	int idevice, int ithread, void* arg)
{
	// Unpack my personal config structure.
	config_t* config = (config_t*)arg + idevice;
	
	size_t size = config->nx * config->ny * sizeof(float);

	// Create device arrays for input and output data.
	coaccel_address addr = coaccel_device_malloc(device, size);
	config->in_dev = addr.value;
	addr = coaccel_device_malloc(device, size);
	config->out_dev = addr.value;

	// Memcpy interface in comparison to regular memcpy has more
	// common form:
	// 1) API is *always* async
	// 2) Transfer details are described with reusable memory request (memreq).

	// Allocate and fill memory request.
	coaccel_memreq memreq = coaccel_memreq_allocate(1, 1);
	coaccel_memreq_query(memreq,
		COACCEL_ADDRESS, config->in_dev,	// destination
		COACCEL_ADDRESS, config->inout_cpu,	// source
		1, size, COACCEL_ENDMARK);
	
	// Start memcpy operation. In comparison to CUDA,
	// we don't use HOST / DEVICE terminology. Locations are
	// instead LOCAL and GLOBAL with respect to device where
	// this program is executed.
	coaccel_memcpy desc = coaccel_device_memcpy_start(
		device, memreq, COACCEL_MEMCPY_LOCAL_TO_REMOTE);
	
	// Imidiately request memcpy operation synchronization,
	// like as we do sync mode copy.
        coaccel_device_memcpy_finish(device, desc);
        
        // Release memreq structure.
        coaccel_memreq_dispose(memreq);
        
        printf("Device %d initialized\n", idevice);

	return 0;
}

// Release memory used by each device.
int deinit(
	coaccel_device_group group, coaccel_device device,
	int idevice, int ithread, void* arg)
{
	// Unpack my personal config structure.
	config_t* config = (config_t*)arg + idevice;

	size_t size = config->nx * config->ny * sizeof(float);
	
	// Offload results back to shared memory.
	coaccel_memreq memreq = coaccel_memreq_allocate(1, 1);
	coaccel_memreq_query(memreq,
		COACCEL_ADDRESS, config->inout_cpu,
		COACCEL_ADDRESS, config->in_dev,
		1, size, COACCEL_ENDMARK);
	coaccel_memcpy desc = coaccel_device_memcpy_start(
		device, memreq, COACCEL_MEMCPY_REMOTE_TO_LOCAL);
        coaccel_device_memcpy_finish(device, desc);
        coaccel_memreq_dispose(memreq);

	// Dispose device buffers.
	coaccel_device_free(device, (coaccel_address)(void*)config->in_dev);
	coaccel_device_free(device, (coaccel_address)(void*)config->out_dev);

        printf("Device %d deinitialized\n", idevice);

	return 0;
}

// Compute 2d pattern on each device.
int process(
	coaccel_device_group group, coaccel_device device,
	int idevice, int ithread, void* arg)
{
	// Unpack my personal config structure.
	config_t* config = (config_t*)arg + idevice;

	// Switch callee, depending on the current device mode.
	switch (coaccel_device_get_mode(device))
	{
		case COACCEL_DEVMODE_CUDA_SYNC :
			coaccel_device_lock(device);
			pattern2d_gpu(1, config->nx, 1, 1, config->ny, 1,
				config->in_dev, config->out_dev, idevice);
			coaccel_device_unlock(device);
			break;
		case COACCEL_DEVMODE_CPU_SYNC :
			pattern2d_cpu(1, config->nx, 1, 1, config->ny, 1,
				config->in_dev, config->out_dev, idevice);
	}

	config->step++;
	
	// Swap device input and output buffers.
	float* swap = config->in_dev;
	config->in_dev = config->out_dev;
	config->out_dev = swap;

	printf("Device %d completed step %d\n", idevice, config->step);

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
	
	// Create new empty COACCEL device group.
	coaccel_device_group devices = coaccel_device_group_create();

	// Fill group with GPU devices.
	for (int i = 0; i < ndevices; i++)
	{
		// Initialize COACCEL device supporting CUDA
		// and synchronous memory I/O.
		coaccel_device device = coaccel_device_init(
			argc, argv, COACCEL_DEVMODE_CUDA_SYNC, NULL);
		if (!device)
		{
			fprintf(stderr, "Cannot initialize CUDA device\n");
			return -1;
		}

		// Add created device to group.
		coaccel_device_add(devices, device, i);
	}
	
	// To show hybrid computations, also create
	// CPU device and add it to group.
	coaccel_device device = coaccel_device_init(
		argc, argv, COACCEL_DEVMODE_CPU_SYNC, NULL);
	if (!device)
	{
		fprintf(stderr, "Cannot initialize CPU device\n");
		return -1;
	}
	coaccel_device_add(devices, device, ndevices);
	
	// Create input data. Let each device to have an equal piece
	// of single shared data array.
	size_t np = nx * ny;
	float* inout = (float*)malloc((ndevices + 1) * np * sizeof(float));
	float invdrandmax = 1.0 / RAND_MAX;
	for (size_t i = 0; i < np; i++)
		inout[i] = rand() * invdrandmax;
	for (int i = 0; i < ndevices; i++)
		memcpy(inout + np * (i + 1), inout, np * sizeof(float));

	// Create workers configs. This data will be passed as
	// input to all multi callbacks.	
	config_t* configs = (config_t*)malloc(
		sizeof(config_t) * (ndevices + 1));
	for (int i = 0; i < ndevices + 1; i++)
	{
		configs[i].step = 0;
		configs[i].nx = nx; configs[i].ny = ny;
		configs[i].inout_cpu = inout + np * i;
	}

	// Initialize threaded execution.
	coaccel_multi multi = coaccel_multi_init(
		devices, 1, &init, (void*)configs);
	if (!multi)
	{
		fprintf(stderr, "Cannot initialize COACCEL multi\n");
		return -1;
	}

	// Perform several steps of threaded execution.
	for (int i = 0; i < nticks; i++)
		coaccel_multi_step_all(multi, process, (void*)configs);

	// Finalize threaded execution.
	coaccel_multi_finalize(multi, &deinit, (void*)configs);

	// Dispose devices and group.
	for (int i = 0; i < ndevices + 1; i++)
	{
		coaccel_device_dispose(
			coaccel_device_get(devices, i));
	}
	coaccel_device_group_dispose(devices);
	
	// Compare each GPU result to CPU result.
	float* control = inout + ndevices * np;
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

