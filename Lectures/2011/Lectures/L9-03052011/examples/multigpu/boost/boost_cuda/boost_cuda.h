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

#include <boost/thread.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

// The thread management system.
class ThreadRunner
{
	static int status;
	static boost::mutex m;
	
	void SetLastError(int status);
	
	int idevice;
	int step, finish;
	int nx, ny;

	float *in_dev, *out_dev;
	
	CUcontext ctx;

	boost::barrier* b1, b2;
	boost::thread t;

	// The function executed by each thread assigned with CUDA device.
	void thread_func();

public:
	
	ThreadRunner(int idevice, int nx, int ny, boost::barrier* b);

	static int GetLastError();

	// Load input data on device.
	void Load(float* input);

	// Unload output data from device.
	void Unload(float* output);
	
	void Pass();

	~ThreadRunner();
};

