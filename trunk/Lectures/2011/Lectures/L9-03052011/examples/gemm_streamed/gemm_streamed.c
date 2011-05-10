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
 * Based on original sample by Everett Philips.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas.h>
#include <cuda_runtime.h>

#define CUBLAS_ERR_CHECK(message) \
if (cublasGetError() != CUBLAS_STATUS_SUCCESS) \
{ \
	fprintf(stderr, "CUBLAS: %s:%d - %s \n", \
		__FILE__, __LINE__, message); \
	return EXIT_FAILURE; \
}

#define CUDA_ERR_CHECK(message) \
{ \
	cudaError_t cuda_error; \
	cuda_error = cudaGetLastError(); \
	if ( cuda_error != cudaSuccess) \
	{ \
		fprintf(stderr, "CUDA: %s:%d - %s - %s \n", \
		__FILE__, __LINE__, cudaGetErrorString(cuda_error), message); \
		return EXIT_FAILURE; \
	} \
}

#define HAVE_SINGLE
#include "gemm_streamed.h"
#undef HAVE_SINGLE

#define HAVE_DOUBLE
#include "gemm_streamed.h"
#undef HAVE_DOUBLE

int main(int argc, char* argv[])
{
	if (argc != 10)
	{
		printf("Usage: %s <precision> %s %s\n", argv[0],
			"<n_min> <n_max> <n_step> <transa> <transb>",
			"<alpha> <beta> <n_streams>");
		return 0;
	}

	int precision = atoi(argv[1]);
	assert((precision == 4) || (precision == 8));

	int n_min = atoi(argv[2]); assert(n_min > 0);
	int n_max = atoi(argv[3]);
	int n_step = atoi(argv[4]); assert(n_step > 0);
	
	char transa = argv[5][0];
	assert((transa == 'n') || (transa == 'N') ||
		(transa == 't') || (transa == 'T'));
	char transb = argv[6][0];
	assert((transb == 'n') || (transb == 'N') ||
		(transb == 't') || (transb == 'T'));

	int n_streams = atoi(argv[9]); assert(n_streams > 0);

	printf("n\ttime\t\tgflops\t\ttest\tenorm\t\trnorm\n");

	if (precision == 4)
	{
		float alpha = (float)atof(argv[7]);
		float beta = (float)atof(argv[8]);

		for (int n = n_min; n < n_max; n += n_step)
		{
			sgemm_serial(transa, transb, alpha, beta, n);
			sgemm_streamed(transa, transb, alpha, beta, n, n_streams);
		}
	}
	
	if (precision == 8)
	{

		double alpha = atof(argv[7]);
		double beta = atof(argv[8]);

		for (int n = n_min; n < n_max; n += n_step)
		{
			dgemm_serial(transa, transb, alpha, beta, n);
			dgemm_streamed(transa, transb, alpha, beta, n, n_streams);
		}
	}
}

