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

#include <errno.h>
#include <malloc.h>
#include <stdio.h>
#include <unistd.h>

int nticks = 10;

// The size of memory region.
size_t szmem = 1024;

int main(int argc, char* argv[])
{
	// Allocate array.
	char* data = (char*)malloc(szmem);
	sprintf(data, "Initial data array state");

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
	int master = fork_status ? 1 : 0, worker = !master;

	// Get the process ID.
	int pid = (int)getpid();
	
	// Perform some "iterations" on data array private to each process.
	for (int i = 0; i < nticks; i++)
	{
		// We make sure data always contains content initially derived
		// from parent process or changed by current process on previous
		// iteration.
		printf("\n>> %s\n", data);
		
		// Change data content.
		sprintf(data, "Process %d works on data %p, tick %d", pid, data, i);
		
		// Print changed content.
		printf("<< %s\n\n", data);
		
		// Emulate difference in processing speed:
		// child iterates 2 times faster than parent.
		sleep(2 * master + 1 * worker);
	}
	
	free(data);
	
	return 0;
}
