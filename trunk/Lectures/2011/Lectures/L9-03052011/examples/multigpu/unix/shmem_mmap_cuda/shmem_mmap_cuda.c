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
 * Each process works on shared data in critical section and own private data.
 *
 */

#include "pattern2d.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
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
	float *inout_cpu, *in_dev, *out_dev;
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

	// Create shared memory region.
	int fd = shm_open("/shmem_mmap_cuda_shm",
		O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
	if (fd == -1)
	{
		fprintf(stderr, "Cannot open shared region, errno = %d\n", errno);
		return errno;
	}

	// Create first semaphore (set to 0 to create it initially locked).
	sem_t* sem1 = sem_open("/shmem_mmap_cuda_sem1",
		O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 0);
	if (sem1 == SEM_FAILED)
	{
		fprintf(stderr, "Cannot open semaphore #1, errno = %d\n", errno);
		return errno;
	}

	// Create second semaphore (set to 0 to create it initially locked).
	sem_t* sem2 = sem_open("/shmem_mmap_cuda_sem2",
		O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 0);
	if (sem2 == SEM_FAILED)
	{
		fprintf(stderr, "Cannot open semaphore #2, errno = %d\n", errno);
		return errno;
	}

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

	// Get the process ID.
	int pid = (int)getpid();

	// By fork return value we can determine the process role:
	// master or child (worker).
	int master = fork_status ? 1 : 0, worker = !master;

	int ndevices = 0;
	cudaError_t cuda_status = cudaGetDeviceCount(&ndevices);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot get the cuda device count by process %d, status = %d\n",
			pid, cuda_status);
		return cuda_status;
	}
	
	// Return if no cuda devices present.
	if (master)
		printf("%d CUDA device(s) found\n", ndevices);
	if (!ndevices) return 0;
	ndevices = 1;

	size_t np = nx * ny;
	size_t size = np * sizeof(float);

	float* inout;
	
	if (!master)
	{
		// Lock semaphore to finish shared region configuration on master.
		int sem_status = sem_wait(sem1);
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot wait on semaphore by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}

		// Map the shared region into the address space of the current process.
		inout = (float*)mmap(0, size * (ndevices + 1),
			PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (inout == MAP_FAILED)
		{
			fprintf(stderr, "Cannot map shared region to memory by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}
	else
	{
		config.idevice = ndevices;

		// Set shared region size.
		int ftrunk_status = ftruncate(fd, size * (ndevices + 1));
		if (ftrunk_status == -1)
		{
			fprintf(stderr, "Cannot truncate shared region, errno = %d\n", errno);
			return errno;
		}

		// Map the shared region into the address space of the current process.
		inout = (float*)mmap(0, size * (ndevices + 1),
			PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (inout == MAP_FAILED)
		{
			fprintf(stderr, "Cannot map shared region to memory by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}

		// Create input data. Let each device to have an equal piece
		// of single shared data array.
		float invdrandmax = 1.0 / RAND_MAX;
		for (size_t i = 0; i < np; i++)
			inout[i] = rand() * invdrandmax;
		for (int i = 0; i < ndevices; i++)
			memcpy(inout + np * (i + 1), inout, np * sizeof(float));

		// Sync changed content with shared region.
		int msync_status = msync(inout, size * (ndevices + 1), MS_SYNC);
		if (msync_status == -1)
		{
			fprintf(stderr, "Cannot sync shared memory %p, errno = %d\n",
				inout, errno);
			return errno;
		}

		// Unlock semaphore to let other processes to move forward.
		int sem_status = sem_post(sem1);
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot post on semaphore by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}

	config.inout_cpu = inout + config.idevice * np;

	// Let workers to use CUDA devices, and master - the CPU.
	// Create device buffers.
	if (worker)
	{
		// Create device arrays for input and output data.
		cuda_status = cudaMalloc((void**)&config.in_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate CUDA input buffer by process %d, status = %d\n",
				pid, cuda_status);
			return cuda_status;
		}
		cuda_status = cudaMalloc((void**)&config.out_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate CUDA output buffer by process %d, status = %d\n",
				pid, cuda_status);
			return cuda_status;
		}
	}
	else
	{
		// Create device arrays for input and output data.
		config.in_dev = (float*)malloc(size);
		config.out_dev = (float*)malloc(size);
	}

	printf("Device %d initialized py process %d\n", config.idevice, pid);

	// Perform some "iterations" on data arrays, assigned to devices,
	// and shift input data array after each iteration.
	for (int i = 0; i < nticks; i++)
	{
		int status;
		if (master)
		{
			// Copy input data to device buffer.
			memcpy(config.in_dev, config.inout_cpu, size);

			status = pattern2d_cpu(1, config.nx, 1, 1, config.ny, 1,
				config.in_dev, config.out_dev, config.idevice);
			if (status)
			{
				fprintf(stderr, "Cannot execute pattern 2d by process %d, status = %d\n",
					pid, status);
				return status;
			}

			// Copy output data from device buffer.
			memcpy(config.inout_cpu, config.out_dev, size);

			// Sync with changed content in shared region.
			int msync_status = msync(inout, size * (ndevices + 1), MS_SYNC);
			if (msync_status == -1)
			{
				fprintf(stderr, "Cannot sync shared memory %p, errno = %d\n",
					inout, errno);
				return errno;
			}

			int sem_status = sem_post(sem1);
			if (sem_status == -1)
			{
				fprintf(stderr, "Cannot post on semaphore #1 by process %d, errno = %d\n",
					pid, errno);
				return errno;
			}	

			sem_status = sem_wait(sem2);
			if (sem_status == -1)
			{
				fprintf(stderr, "Cannot post on semaphore #2 by process %d, errno = %d\n",
					pid, errno);
				return errno;
			}	
		}
		else
		{
			// Copy input data to device buffer.
			cuda_status = cudaMemcpy(config.in_dev, config.inout_cpu, size,
				cudaMemcpyHostToDevice);
			if (cuda_status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy input data to CUDA buffer by process %d, status = %d\n",
					pid, cuda_status);
				return cuda_status;
			}

			status = pattern2d_gpu(1, config.nx, 1, 1, config.ny, 1,
				config.in_dev, config.out_dev, config.idevice);
			if (status)
			{
				fprintf(stderr, "Cannot execute pattern 2d by process %d, status = %d\n",
					pid, status);
				return status;
			}

			// Copy output data from device buffer.
			cuda_status = cudaMemcpy(config.inout_cpu, config.out_dev, size,
				cudaMemcpyDeviceToHost);
			if (cuda_status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy output data from CUDA buffer by process %d, status = %d\n",
					pid, cuda_status);
				return cuda_status;
			}

			// Sync with changed content in shared region.
			int msync_status = msync(inout, size * (ndevices + 1), MS_SYNC);
			if (msync_status == -1)
			{
				fprintf(stderr, "Cannot sync shared memory %p, errno = %d\n",
					inout, errno);
				return errno;
			}
			
			int sem_status = sem_wait(sem1);
			if (sem_status == -1)
			{
				fprintf(stderr, "Cannot wait on semaphore #1 by process %d, errno = %d\n",
					pid, errno);
				return errno;
			}			

			sem_status = sem_post(sem2);
			if (sem_status == -1)
			{
				fprintf(stderr, "Cannot post on semaphore #2 by process %d, errno = %d\n",
					pid, errno);
				return errno;
			}
		}

		// At this point two processes are synchronized.

		config.step++;
		
		// Reassign porcesses' input data segments to show some
		// possible manipulation on shared memory.
		// Here we perform cyclic shift of data pointers.
		config.idevice++;
		config.idevice %= ndevices + 1;
		config.inout_cpu = inout +  config.idevice * np;
	}

	// Release device buffers.
	if (worker)
	{
		cuda_status = cudaFree(config.in_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot release input buffer by process %d, status = %d\n",
				pid, cuda_status);
			return cuda_status;
		}
		cuda_status = cudaFree(config.out_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot release output buffer by process %d, status = %d\n",
				pid, cuda_status);
			return cuda_status;
		}
	}
	else
	{
		free(config.in_dev);
		free(config.out_dev);
	}
	
	printf("Device %d deinitialized py process %d\n", config.idevice, pid);

	// On master process perform results check:
	// compare each GPU result to CPU result.
	if (master)
	{
		float* control = inout + np * ndevices;
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
	}
	
	// Unlink semaphore.
	if (master)
	{
		int sem_status = sem_unlink("/shmem_mmap_cuda_sem1");
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot unlink semaphore #1 by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}
	
	// Close semaphore.
	int sem_status = sem_close(sem1);
	if (sem_status == -1)
	{
		fprintf(stderr, "Cannot close semaphore #1 by process %d, errno = %d\n",
			pid, errno);
		return errno;
	}

	// Unlink semaphore.
	if (master)
	{
		int sem_status = sem_unlink("/shmem_mmap_cuda_sem2");
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot unlink semaphore #2 by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}
	
	// Close semaphore.
	sem_status = sem_close(sem2);
	if (sem_status == -1)
	{
		fprintf(stderr, "Cannot close semaphore #2 by process %d, errno = %d\n",
			pid, errno);
		return errno;
	}

	// Unmap shared region.
	close(fd);
	int munmap_status = munmap(inout, size * (ndevices + 1));
	if (munmap_status == -1)
	{
		fprintf(stderr, "Cannot unmap shared region by process %d, errno = %d\n",
			pid, errno);
		return errno;
	}
	
	// Unlink shared region.
	if (master)
	{
		int unlink_status = shm_unlink("/shmem_mmap_cuda_shm");
		if (unlink_status == -1)
		{
			fprintf(stderr, "Cannot unlink shared region by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}
	
	return 0;
}
