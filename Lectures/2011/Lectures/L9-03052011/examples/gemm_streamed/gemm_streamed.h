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

#ifdef HAVE_SINGLE
#define real float
#define blas_gemm sgemm_
#define cublas_gemm cublasSgemm
#define gemm_serial sgemm_serial
#define gemm_streamed sgemm_streamed
#define generate_data sgenerate_data
#define tolerance stolerance
#endif

#ifdef HAVE_DOUBLE
#define real double
#define blas_gemm dgemm_
#define cublas_gemm cublasDgemm
#define gemm_serial dgemm_serial
#define gemm_streamed dgemm_streamed
#define generate_data dgenerate_data
#define tolerance dtolerance
#endif

void blas_gemm(char* transa, char* transb, int* m, int* n, int* k,
	real* alpha, real* A, int* lda, real* B, int* ldb,
	real* beta, real* C, int* ldc);

// The control and production results maximum
// allowed difference.
const real tolerance = 1e-6;

int generate_data(int n, real* A, real* B, real* C)
{
	// Fill the matrices with test data
	real invrandmax = 1.0 / (real)RAND_MAX;
	for (int i = 0; i < n; i++)
	{
		A[i] = rand() * invrandmax;
		B[i] = rand() * invrandmax;
	}
}

int gemm_serial(char transa, char transb, real alpha, real beta, int n)
{
	printf("%d\t", n); fflush(stdout);

	int n2 = n * n;

	// Initialize CUBLAS
	cublasStatus status = cublasInit();
	assert(status == CUBLAS_STATUS_SUCCESS);

	// Allocate host memory for the matrices
	real* h_A = (real*)malloc(n2 * sizeof(real)); assert(h_A);
	real* h_B = (real*)malloc(n2 * sizeof(real)); assert(h_B);
	real* h_C = (real*)malloc(n2 * sizeof(real)); assert(h_C);
	real* h_C_ref = (real*)malloc(n2 * sizeof(real)); assert(h_C_ref);

	generate_data(n2, h_A, h_B, h_C);

	// Allocate device memory for the matrices.
	real* d_A; status = cublasAlloc(n2, sizeof(real), (void**)&d_A);
	assert(status == CUBLAS_STATUS_SUCCESS);
	real* d_B; status = cublasAlloc(n2, sizeof(real), (void**)&d_B);
	assert(status == CUBLAS_STATUS_SUCCESS);
	real* d_C; status = cublasAlloc(n2, sizeof(real), (void**)&d_C);
	assert(status == CUBLAS_STATUS_SUCCESS);

	cudaEvent_t start; cudaEventCreate(&start); 
	cudaEventRecord(start, 0);
	
	// Initialize the device matrices with the host matrices
	status = cublasSetVector(n2, sizeof(real), h_A, 1, d_A, 1);
	assert(status == CUBLAS_STATUS_SUCCESS);
	status = cublasSetVector(n2, sizeof(real), h_B, 1, d_B, 1);
	assert(status == CUBLAS_STATUS_SUCCESS);
	status = cublasSetVector(n2, sizeof(real), h_C, 1, d_C, 1);
	assert(status == CUBLAS_STATUS_SUCCESS);

	// Perform matmul using CUBLAS
	cublas_gemm(transa, transb, n, n, n, 
		alpha, d_A, n, d_B, n, beta, d_C, n);
	status = cublasGetError();
	assert(status == CUBLAS_STATUS_SUCCESS);
	
	// Read the result back
	status = cublasGetVector(n2, sizeof(real), d_C, 1, h_C, 1);
	assert(status == CUBLAS_STATUS_SUCCESS);

	cudaEvent_t stop; cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	float timer_ev; cudaEventElapsedTime(&timer_ev, start, stop);
	double dn = (double)n, gflops = 2.0e-6 * dn * dn * dn / (double)timer_ev;
	printf("%f sec\t%f\t", timer_ev / 1000.0, gflops); fflush(stdout);

	// Perform matmul using host BLAS
	blas_gemm(&transa, &transb, &n, &n, &n, 
		&alpha, h_A, &n, h_B, &n, &beta, h_C_ref, &n);
	
	// Check result against reference
	real error_norm = 0, ref_norm = 0;
	for (int i = 0; i < n2; i++)
	{
		real diff = h_C_ref[i] - h_C[i];
		error_norm += diff * diff;
		ref_norm += h_C_ref[i] * h_C_ref[i];
	}

	// Release host memory
	if (h_A) free(h_A);
	if (h_B) free(h_B);
	if (h_C) free(h_C);
	if (h_C_ref) free(h_C_ref);

	// Release device memory.
	status = cublasFree(d_A); assert(status == CUBLAS_STATUS_SUCCESS);
	status = cublasFree(d_B); assert(status == CUBLAS_STATUS_SUCCESS);
	status = cublasFree(d_C); assert(status == CUBLAS_STATUS_SUCCESS);

	status = cublasShutdown();
	assert(status == CUBLAS_STATUS_SUCCESS);

	error_norm = (real)sqrt((double)error_norm);
	ref_norm = (float) sqrt((double) ref_norm);
	if (fabs(ref_norm) < tolerance)
	{
		printf("Reference norm is 0\n");
		return EXIT_FAILURE;
	}
	printf("%s\t%f\t%f\n", (error_norm / ref_norm < tolerance) ?
		"PASSED" : "FAILED", error_norm, ref_norm);
	fflush(stdout);

	return EXIT_SUCCESS;
}

int gemm_streamed(char transa, char transb, real alpha, real beta, int n, int nstreams)
{
	printf("%d\t", n); fflush(stdout);

	int n2 = n * n;

	// Initialize CUBLAS
	cublasStatus status = cublasInit();
	assert(status == CUBLAS_STATUS_SUCCESS);

	// Allocate host memory for the matrices using the
	// cudaHostAlloc functions needed by the Async Interface
	cudaError_t cudaerr;
	real* h_A; cudaerr = cudaMallocHost((void **)&h_A, n2 * sizeof(real));
	assert(cudaerr == cudaSuccess);
	real* h_B; cudaerr = cudaMallocHost((void **)&h_B, n2 * sizeof(real));
	assert(cudaerr == cudaSuccess);
	real* h_C; cudaerr = cudaMallocHost((void **)&h_C, n2 * sizeof(real));
	assert(cudaerr == cudaSuccess);
	real* h_C_ref = (real*)malloc(n2 * sizeof(real)); assert(h_C_ref);

	generate_data(n2, h_A, h_B, h_C);

	// Allocate data structures
	cudaStream_t* stream = (cudaStream_t*)malloc(
		nstreams * sizeof(cudaStream_t));
	assert(stream);

	cudaEvent_t* event_start = (cudaEvent_t*)malloc(
		nstreams * sizeof(cudaEvent_t));
	assert(event_start);

	cudaEvent_t* event_end = (cudaEvent_t*)malloc(
		nstreams * sizeof(cudaEvent_t));
  	assert(event_end);

	real** d_B = (real**)malloc(nstreams * sizeof(real*)); assert(d_B);
	real** d_C = (real**)malloc(nstreams * sizeof(real*)); assert(d_C);

	// Create a start and end event for each stream
	for (int i = 0; i < nstreams; i++)
	{
		cudaerr = cudaStreamCreate(&stream[i]);
		assert(cudaerr == cudaSuccess);
		cudaerr = cudaEventCreate(&event_start[i]);
		assert(cudaerr == cudaSuccess);
		cudaerr = cudaEventCreate(&event_end[i]);
		assert(cudaerr == cudaSuccess);
	}
	
	// Allocate device memory for the matrices
	real* d_A; cudaerr = cublasAlloc(n2, sizeof(real), (void**)&d_A);
	assert(cudaerr == cudaSuccess);

	// Allocate host memory for reading back the result
	// from device memory
	cudaerr = cudaMallocHost((void**)&h_C, n2 * sizeof(real));
	assert(cudaerr == cudaSuccess);

	for (int istream = 0; istream < nstreams; istream++)
	{
		int szpart = n / nstreams;
		size_t shift = szpart * istream;
		if (istream == nstreams - 1)
			szpart += n % nstreams;
		
		status = cublasAlloc(n * szpart, sizeof(real), (void**)&d_B[istream]);
		assert(status == CUBLAS_STATUS_SUCCESS);		
		status = cublasAlloc(n * szpart, sizeof(real), (void**)&d_C[istream]);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	cudaEventRecord(event_start[0], 0);
	
	// Initialize the device matrices with the
	// host matrices using the async interface
	status = cublasSetVector(n2, sizeof(real),
		h_A, 1, d_A, 1);
	assert(status == CUBLAS_STATUS_SUCCESS);

	for (int istream = 0; istream < nstreams; istream++)
	{
		int szpart = n / nstreams;
		size_t shift = n * szpart * istream;
		if (istream == nstreams - 1)
			szpart += n % nstreams;
		
		status = cublasSetVectorAsync(n * szpart, sizeof(real),
			h_B + shift, 1, d_B[istream], 1, stream[istream]);
		assert(status == CUBLAS_STATUS_SUCCESS);
		status = cublasSetVectorAsync(n * szpart, sizeof(real),
			h_C + shift, 1, d_C[istream], 1, stream[istream]);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	for (int istream = 0; istream < nstreams; istream++)
	{
		int szpart = n / nstreams;
		if (istream == nstreams - 1)
			szpart += n % nstreams;
		
		// Setup async operations
		status = cublasSetKernelStream(stream[istream]);
		assert(status == CUBLAS_STATUS_SUCCESS);

		// Perform matmul using CUBLAS
		cublas_gemm(transa, transb, n, szpart, n,
			alpha, d_A, n, d_B[istream], n, beta, d_C[istream], n);
		status = cublasGetError();
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	// Sync all
	for (int istream = 0; istream < nstreams; istream++)
	{
		int szpart = n / nstreams;
		size_t shift = n * szpart * istream;
		if (istream == nstreams - 1)
			szpart += n % nstreams;

		// Read the result back
		status = cublasGetVectorAsync(n * szpart, sizeof(real),
			d_C[istream], 1, h_C + shift, 1, stream[istream]);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	cudaEventRecord(event_end[0], 0);
	cudaEventSynchronize(event_end[0]);
	
	float timer_ev; cudaEventElapsedTime(&timer_ev, event_start[0], event_end[0]);
	double dn = (double)n, gflops = 2.0e-6 * dn * dn * dn / (double)timer_ev;
	printf("%f sec\t%f\t", timer_ev / 1000.0, gflops); fflush(stdout);

	// Perform matmul using host BLAS
	blas_gemm(&transa, &transb, &n, &n, &n, 
		&alpha, h_A, &n, h_B, &n, &beta, h_C_ref, &n);

	// Check result against reference
	real error_norm = 0, ref_norm = 0;
	for (int i = 0; i < n2; i++)
	{
		real diff = h_C_ref[i] - h_C[i];
		error_norm += diff * diff;
		ref_norm += h_C_ref[i] * h_C_ref[i];
	}

	cudaerr = cudaFreeHost(h_A); assert(cudaerr == cudaSuccess);
	cudaerr = cudaFreeHost(h_B); assert(cudaerr == cudaSuccess);
	cudaerr = cudaFreeHost(h_C); assert(cudaerr == cudaSuccess);
	if (h_C_ref) free(h_C_ref);

	cudaerr = cublasFree(d_A); assert(cudaerr == cudaSuccess);

	for (int istream = 0; istream < nstreams; istream++)
	{
		cudaerr = cudaStreamDestroy(stream[istream]);
		assert(cudaerr == cudaSuccess);
		cudaerr = cublasFree(d_B[istream]);
		assert(cudaerr == cudaSuccess);
		cublasFree(d_C[istream]);
		assert(cudaerr == cudaSuccess);
	}
		
	status = cublasShutdown();
	assert(status == CUBLAS_STATUS_SUCCESS);

	error_norm = (real)sqrt((double)error_norm);
	ref_norm = (real)sqrt((double)ref_norm);
	if (fabs(ref_norm) < tolerance)
	{
		printf("Reference norm is 0\n");
		return EXIT_FAILURE;
	}
	printf("%s\t%f\t%f\n", (error_norm / ref_norm < tolerance) ?
		"PASSED" : "FAILED", error_norm, ref_norm);
	fflush(stdout);

	return EXIT_SUCCESS;
}

#undef real
#undef blas_gemm
#undef cublas_gemm
#undef gemm_serial
#undef gemm_streamed
#undef generate_data
#undef tolerance
