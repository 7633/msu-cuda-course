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
 * This sample demonstates parallel execution using
 * Message Passing Interface (MPI). Each process works on private
 * data and synchronizes with master after all iterations.
 *
 */

#include "pattern2d.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// The thread configuration structure.
typedef struct
{
	int idevice;
	int step;
	int nx, ny;
	float *in_cpu, *out_cpu;
	float *in_dev, *out_dev;
}
config_t;

int nticks = 10;

// The size of memory region.
int nx = 128, ny = 128;

int main(int argc, char* argv[])
{
	// Process config (to be filled completely
	// later).
	config_t config;
	config.idevice = 0;
	config.nx = nx;
	config.ny = ny;
	config.step = 0;

	// Initialize MPI. From this point the specified
	// number of processes will be executed in parallel.
	int mpi_status = MPI_Init(&argc, &argv);
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize MPI, status = %d\n", mpi_status);
		return mpi_status;
	}
	
	// Get the size of the MPI global communicator,
	// that is get the total number of MPI processes.
	int nprocesses;
	mpi_status = MPI_Comm_size(MPI_COMM_WORLD, &nprocesses);
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot retrieve the number of MPI processes, status = %d\n",
			mpi_status);
		return mpi_status;
	}
	
	// Get the rank (index) of the current MPI process
	// in the global communicator.
	mpi_status = MPI_Comm_rank(MPI_COMM_WORLD, &config.idevice);
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot retrieve the rank of current MPI process, status = %d\n",
			mpi_status);
		return mpi_status;
	}

	int ndevices = 0;
	cudaError_t cuda_status = cudaGetDeviceCount(&ndevices);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot get the cuda device count by process %d, status = %d\n",
			config.idevice, cuda_status);
		return cuda_status;
	}

	// Return if no cuda devices present.
	if (config.idevice == 0)
		printf("%d CUDA device(s) found\n", ndevices);
	if (!ndevices) return 0;
	ndevices = 1;

	// Let the last process to be master.
	int master = (config.idevice == ndevices), worker = !master;
	
	// Continue running only ndevices + 1 processes,
	// let all others to finish now.
	if (config.idevice > ndevices)
	{
		mpi_status = MPI_Finalize();
		if (mpi_status != MPI_SUCCESS)
		{
			fprintf(stderr, "Cannot finalize MPI, status = %d\n",
				mpi_status);
			return mpi_status;
		}

		return 0;
	}
	
	// TODO: create a new communicator instead of MPI_COMM_WORLD
	// to send/recv messages only between working processes.

	size_t np = nx * ny;
	size_t size = np * sizeof(float);

	// Create host arrays for input and output data.
	config.in_cpu = (float*)malloc(size);
	config.out_cpu = (float*)malloc(size);
	
	if (master)
	{
		// Create input data.
		float invdrandmax = 1.0 / RAND_MAX;
		for (size_t i = 0; i < np; i++)
			config.in_cpu[i] = rand() * invdrandmax;
	}

	// Let each device to have equal dataset
	// in its private array.
	mpi_status = MPI_Bcast(config.in_cpu, np, MPI_FLOAT, ndevices, 
		MPI_COMM_WORLD);
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot broadcast input data by process %d, status = %d\n",
			mpi_status);
		return mpi_status;
	}

	// Let workers to use CUDA devices, and master - the CPU.
	// Create device buffers.
	if (worker)
	{
		// Create device arrays for input and output data.
		cuda_status = cudaMalloc((void**)&config.in_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate CUDA input buffer by process %d, status = %d\n",
				config.idevice, cuda_status);
			return cuda_status;
		}
		cuda_status = cudaMalloc((void**)&config.out_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate CUDA output buffer by process %d, status = %d\n",
				config.idevice, cuda_status);
			return cuda_status;
		}
	}
	else
	{
		// Create device arrays for input and output data.
		config.in_dev = (float*)malloc(size);
		config.out_dev = (float*)malloc(size);
	}

	printf("Device %d initialized\n", config.idevice);

	// Perform some "iterations" on data arrays, assigned to devices,
	// and exchange arrays after each iteration.
	for (int i = 0; i < nticks; i++)
	{
		int status;
		if (master)
		{
			// Copy input data to device buffer.
			memcpy(config.in_dev, config.in_cpu, size);

			status = pattern2d_cpu(1, config.nx, 1, 1, config.ny, 1,
				config.in_dev, config.out_dev, config.idevice);
			if (status)
			{
				fprintf(stderr, "Cannot execute pattern 2d by process %d, status = %d\n",
					config.idevice, status);
				return status;
			}

			// Copy output data from device buffer.
			memcpy(config.out_cpu, config.out_dev, size);
		}
		else
		{
			// Copy input data to device buffer.
			cuda_status = cudaMemcpy(config.in_dev, config.in_cpu, size,
				cudaMemcpyHostToDevice);
			if (cuda_status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy input data to CUDA buffer by process %d, status = %d\n",
					config.idevice, cuda_status);
				return cuda_status;
			}

			status = pattern2d_gpu(1, config.nx, 1, 1, config.ny, 1,
				config.in_dev, config.out_dev, config.idevice);
			if (status)
			{
				fprintf(stderr, "Cannot execute pattern 2d by process %d, status = %d\n",
					config.idevice, status);
				return status;
			}

			// Copy output data from device buffer.
			cuda_status = cudaMemcpy(config.out_cpu, config.out_dev, size,
				cudaMemcpyDeviceToHost);
			if (cuda_status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy output data from CUDA buffer by process %d, status = %d\n",
					config.idevice, cuda_status);
				return cuda_status;
			}
		}

		config.step++;
		
		// Swap input and output buffers.
		float* swap = config.in_cpu;
		config.in_cpu = config.out_cpu;
		config.out_cpu = swap;
	}

	// Release device buffers.
	if (worker)
	{
		cuda_status = cudaFree(config.in_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot release input buffer by process %d, status = %d\n",
				config.idevice, cuda_status);
			return cuda_status;
		}
		cuda_status = cudaFree(config.out_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot release output buffer by process %d, status = %d\n",
				config.idevice, cuda_status);
			return cuda_status;
		}
	}
	else
	{
		free(config.in_dev);
		free(config.out_dev);
	}
	
	printf("Device %d deinitialized\n", config.idevice);

	// On master process perform results check:
	// compare each GPU result to CPU result.
	if (master)
	{
		float* output = config.out_cpu;
		float* control = config.in_cpu;
		for (int idevice = 0; idevice < ndevices; idevice++)
		{
			// Receive output from each worker device.
			mpi_status = MPI_Recv(output, np, MPI_FLOAT, idevice, 0,
				MPI_COMM_WORLD, NULL);
			if (mpi_status != MPI_SUCCESS)
			{
				fprintf(stderr, "Cannot receive output from device %d, status = %d\n",
					idevice, mpi_status);
				return mpi_status;
			}
		
			// Find the maximum abs difference.
			int maxi = 0, maxj = 0;
			float maxdiff = fabs(control[0] - output[0]);
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					float diff = fabs(
						control[i + j * nx] -
						output[i + j * nx]);
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
	}
	else
	{
		// Send worker output to master for check.
		MPI_Send(config.in_cpu, np, MPI_FLOAT, ndevices, 0,
			MPI_COMM_WORLD); 
		if (mpi_status != MPI_SUCCESS)
		{
			fprintf(stderr, "Cannot send output from device %d, status = %d\n",
				config.idevice, mpi_status);
			return mpi_status;
		}
	}

	free(config.in_cpu);
	free(config.out_cpu);
	
	mpi_status = MPI_Finalize();
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot finalize MPI, status = %d\n",
			mpi_status);
		return mpi_status;
	}
	
	return 0;
}
