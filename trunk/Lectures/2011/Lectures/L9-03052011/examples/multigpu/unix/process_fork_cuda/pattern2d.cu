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
 */

#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Maximum allowed results difference.
#define EPS 1e-04

// Grid block size (see comment in pattern2d below).
#define BLOCK_LENGTH 32
#define BLOCK_HEIGHT 16

#define IN(i,j) in[i + (j) * nx]
#define OUT(i,j) out[i + (j) * nx]

// GPU device kernel.
__global__ void pattern2d_gpu(
	int bx, int nx, int ex, int by, int ny, int ey,
	float* in, float* out)
{
	// Compute absolute (i,j) indexes for
	// the current GPU thread using grid mapping params.
	int i = blockIdx.x * BLOCK_LENGTH + threadIdx.x + bx;
	int j = blockIdx.y * BLOCK_HEIGHT + threadIdx.y + by;
	
	// Compute one data point - a piece of
	// work for the current GPU thread.
	OUT(i,j) = sqrtf(fabs(IN(i,j) + IN(i-1,j) + IN(i+1,j) -
		2.0f * IN(i,j-1) + 3.0f * IN(i,j+1)));
}

// CPU control implementation.
void pattern2d_cpu(
	int bx, int nx, int ex, int by, int ny, int ey,
	float* in, float* out)
{
	for (int j = by; j < ny - ey; j++)
		for (int i = bx; i < nx - ex; i++)
			OUT(i,j) = sqrtf(fabs(IN(i,j) + IN(i-1,j) + IN(i+1,j) -
				2.0f * IN(i,j-1) + 3.0f * IN(i,j+1)));
}

// Perform some dummy 2D field processing on GPU and CPU,
// and compare results.
int pattern2d(int nx, int ny, float* in, float* out, int pid, int step)
{
	if ((nx <= 0) || (ny <= 0)) return -1;

	long np = nx * ny;

	size_t size = sizeof(float) * np;

	// Create GPU data array and copy input data to it.
	float* in_gpu;
	cudaError_t status = cudaMalloc((void**)&in_gpu, size);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "%s %d, status = %d\nInsufficient GPU memory?\n",
			"Cannot malloc input memory on GPU by process", pid, status);
		return status;
	}
	status = cudaMemcpy(in_gpu, in, size, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from host to gpu by process %d, status = %d\n",
			pid, status);
		return status;
	}
	
	// Create CPU data output array and get
	// control result using CPU function.
	float* control = (float*)malloc(size);
	memset(control, 0, size);
	pattern2d_cpu(
		1, nx, 1, 1, ny, 1, in, control);
	
	// Configure GPU computational grid:
	// nx = nblocks_x * block_length
	// ny = nblocks_y * block_height
	//
	// NOTE: we have degree of freedom in
	// selecting how real problem grid maps onto
	// computational grid. Usually these params
	// are tuned to get optimal performance.
	//
	// NOTE: chose of grid/block config is
	// also limited by device properties:
	// - Maximum number of threads per block (512)
	// - Maximum sizes of each dimension of a block (512 x 512 x 64)
	// - Maximum sizes of each dimension of a grid (65535 x 65535 x 1)
	int nblocks_x = (nx - 2) / BLOCK_LENGTH;
	int nblocks_y = (ny - 2) / BLOCK_HEIGHT;
	
	// Perform the same processing on GPU,
	// returning result to GPU array.
	float* out_gpu;
	status = cudaMalloc((void**)&out_gpu, size);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "%s %d, status = %d\nInsufficient GPU memory?\n",
			"Cannot malloc output memory on GPU by process", pid, status);
		return status;
	}
	status = cudaMemset(out_gpu, 0, size);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot erase output memory on GPU by process %d, status = %d\n",
			pid, status);
		return status; 
	}
	pattern2d_gpu<<<
		dim3(nblocks_x, nblocks_y, 1),
		dim3(BLOCK_LENGTH, BLOCK_HEIGHT, 1)>>>(
			1, nx, 1, 1, ny, 1, in_gpu, out_gpu);
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot execute CUDA kernel by process %d, status = %d\n",
			pid, status);
		return status;
	}
	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize thread by process %d, status = %d\n",
			pid, status);
		return status;
	}
		
	// Copy GPU result from GPU memory to CPU buffer.
	status = cudaMemcpy(out, out_gpu, size, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from gpu to host by process %d, status = %d\n",
			pid, status);
		return status;
	}

	// Don't bother with processing the remainder
	// on GPU. Do it on CPU instead.
	pattern2d_cpu(
		1, nx, 1,
		ny - (ny - 2) % BLOCK_HEIGHT - 2, ny, 1,
		in, out);
	pattern2d_cpu(
		nx - (nx - 2) % BLOCK_LENGTH - 2, nx, 1,
		1, ny, 1,
		in, out);
	
	// Compare results and find the maximum abs difference.
	int maxi = 0, maxj = 0;
	float maxdiff = fabs(out[0] - control[0]);
	float* diffs = (float*)malloc(size);
	memset(diffs, 0, size);
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
		{
			float diff = fabs(
				out[i + j * nx] -
				control[i + j * nx]);
			if (diff > maxdiff)
			{
				maxdiff = diff;
				maxi = i; maxj = j;
			}
			diffs[i + j * nx] = diff;
		}

	// Release data arrays.
	status = cudaFree(in_gpu);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot free device input memory by process %d, status = %d\n",
			pid, status);
		return status;
	}
	free(control);
	status = cudaFree(out_gpu);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot free device output memory by process %d, status = %d\n",
			pid, status);
		return status;
	}
	free(diffs);

	printf("Step %d result abs max diff by process %d = %f @ (%d,%d)\n",
		step, pid, maxdiff, maxi, maxj);
	
	return 0;
}

