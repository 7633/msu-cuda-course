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
 * This sample demonstates parallel execution using process forking.
 * Each process works on own private data.
 *
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <malloc.h>
#include <stdio.h>
#include <unistd.h>

// Perform some dummy 2D field processing on GPU and CPU,
// and compare results.
int pattern2d(int nx, int ny, float* in, float* out, int pid, int step);

int nticks = 10;

// The size of memory region.
int nx = 512, ny = 256;
size_t size = nx * ny * sizeof(float);

int main(int argc, char* argv[])
{
	// Allocate input & output arrays.
	float* input = (float*)malloc(size);
	float* output = (float*)malloc(size);

	// Generate input data array of the
	// specified size.
	long np = nx * ny;
	float invdrandmax = 1.0 / RAND_MAX;
	for (long i = 0; i < np; i++)
		input[i] = rand() * invdrandmax;

	// Call fork to create another process.
	// Standard: "Memory mappings created in the parent
	// shall be retained in the child process."
	pid_t fork_status = fork();

	// From this point two processes are running the same code, if no errors.
	if (fork_status == -1)
	{
		fprintf(stderr, "Cannot fork process, errno = %d\n", errno);
		return errno;
	}

	// By fork return value we can determine the process role:
	// master or child (worker).
	int master = fork_status ? 1 : 0;

	int ndevices = 0;
	cudaError_t cuda_status = cudaGetDeviceCount(&ndevices);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot get the cuda device count, status = %d\n",
			cuda_status);
		return cuda_status;
	}
	
	// Return if no cuda devices present.
	if (master)
		printf("%d CUDA device(s) found\n", ndevices);
	if (!ndevices) return 0;

	// Get the process ID.
	int pid = (int)getpid();
	
	// Use different devices, if more than one present.
	if (ndevices > 1)
	{
		int idevice = 1;
		if (master) idevice = 0;
		
		cuda_status = cudaSetDevice(idevice);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot set CUDA device by process %d, status = %d\n",
				pid, cuda_status);
			return cuda_status;
		}
		printf("Process %d uses device #%d\n", pid, idevice);
	}
	
	// Perform some "iterations" on data array private to each process.
	for (int i = 0; i < nticks; i++)
	{
		// Execute function with CUDA kernel.
		int status = pattern2d(nx, ny, input, output, pid, i);
		if (status)
		{
			fprintf(stderr, "Pattern 2D failed by process %d, status = %d\n",
				pid, status);
			return status;
		}
	}
	
	free(input);
	free(output);
	
	return 0;
}
