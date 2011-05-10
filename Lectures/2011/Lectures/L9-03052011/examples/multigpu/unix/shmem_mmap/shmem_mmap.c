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

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

int nticks = 10;

// The size of memory region.
size_t szmem = 1024;

int main(int argc, char* argv[])
{
	// Allocate and fill the private data array.
	char* private_data = (char*)malloc(szmem);
	sprintf(private_data, "Initial private data array state");

	// Create shared memory region.
	int fd = shm_open("/myshm",
		O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
	if (fd == -1)
	{
		fprintf(stderr, "Cannot open shared region, errno = %d\n", errno);
		return errno;
	}

	// Set shared region size.
	int ftrunk_status = ftruncate(fd, szmem);
	if (ftrunk_status == -1)
	{
		fprintf(stderr, "Cannot truncate shared region, errno = %d\n", errno);
		return errno;
	}

	// Map the shared region into the address space of the master process.
	char* shared_data = (char*)mmap(0, szmem,
		PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (shared_data == MAP_FAILED)
	{
		fprintf(stderr, "Cannot map shared region to memory, errno = %d\n",
			errno);
		return errno;
	}
	
	// Fill shared data array.
	sprintf(shared_data, "Initial shared data array state");

	// Sync changed content with shared region.
	int msync_status = msync(shared_data, szmem, MS_SYNC);
	if (msync_status == -1)
	{
		fprintf(stderr, "Cannot sync shared memory %p, errno = %d\n",
			shared_data, errno);
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
	
	// Create semaphore.
	sem_t* sem = sem_open("/mysem", O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 1);
	if (sem == SEM_FAILED)
	{
		fprintf(stderr, "Cannot open semaphore by process %d, errno = %d\n",
			pid, errno);
		return errno;
	}

	// Perform some "iterations" on data array private to each process.
	for (int i = 0; i < nticks; i++)
	{
		// Lock semaphore to begin working with shared data exclusively.
		int sem_status = sem_wait(sem);
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot wait on semaphore by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	
		// Ensure shared data always contains content was modified by
		// process last time accessed to it.
		printf("\nshared >> %s\n", shared_data);

		// Update shared data content.
		sprintf(shared_data, "Process %d works on data %p, tick %d", pid, shared_data, i);
		
		// Print changed shared content.
		printf("shared << %s\n", shared_data);
	
		// Sync changed content with shared region.
		int msync_status = msync(shared_data, szmem, MS_SYNC);
		if (msync_status == -1)
		{
			fprintf(stderr, "Cannot sync shared memory %p by process %d, errno = %d\n",
				shared_data, pid, errno);
			return errno;
		}

		// Unlock semaphore to finish working with shared data exclusively.
		sem_status = sem_post(sem);
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot post on semaphore by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}

		// Ensure private data always contains content initially derived
		// from parent process or changed by current process on previous
		// iteration.
		printf("\nprivate >> %s\n", private_data);

		// Update private data content.
		sprintf(private_data, "Process %d works on data %p, tick %d", pid, private_data, i);

		// Change private data content.
		sprintf(private_data, "Process %d works on data %p, tick %d", pid, private_data, i);
		
		// Print changed private content.
		printf("private << %s\n", private_data);
		
		// Emulate difference in processing speed:
		// child iterates 2 times faster than parent.
		sleep(2 * master + 1 * worker);
	}
	
	free(private_data);
	
	// Unlink semaphore.
	if (master)
	{
		int sem_status = sem_unlink("/mysem");
		if (sem_status == -1)
		{
			fprintf(stderr, "Cannot unlink semaphore by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}
	
	// Close semaphore.
	int sem_status = sem_close(sem);
	if (sem_status == -1)
	{
		fprintf(stderr, "Cannot close semaphore by process %d, errno = %d\n",
			pid, errno);
		return errno;
	}

	// Unmap shared region.
	close(fd);
	int munmap_status = munmap(shared_data, szmem);
	if (munmap_status == -1)
	{
		fprintf(stderr, "Cannot unmap shared region by process %d, errno = %d\n",
			pid, errno);
		return errno;
	}
	
	// Unlink shared region.
	if (master)
	{
		int unlink_status = shm_unlink("/myshm");
		if (unlink_status == -1)
		{
			fprintf(stderr, "Cannot unlink shared region by process %d, errno = %d\n",
				pid, errno);
			return errno;
		}
	}
	
	return 0;
}
