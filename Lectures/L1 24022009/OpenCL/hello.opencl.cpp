#include <CL\cl.h>

char *program_text =  "int sq(int x) { return x*x; } \
                       __kernel void simple(__global float* a, int iNumElements)  \
												{ \
														int iGID = get_global_id(0); \
														a[iGID] = sq(iGID); \
												}";
const char ** pText = (const char**) &program_text;
size_t N = 512;

int main(int argc, char ** argv)
{
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem    mem;
	cl_device_id * pDeviceId = NULL;
	size_t device_count = 0;

	context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &device_count);

	pDeviceId = new cl_device_id [device_count];

	clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, pDeviceId, NULL);

	command_queue = clCreateCommandQueue(context, pDeviceId[0], 0, 0);

	program = clCreateProgramWithSource(context, 1, pText, NULL, NULL);

	clBuildProgram(program, 0, 0, 0, 0, 0);

	kernel = clCreateKernel(program, "simple", 0);	

	float * pData = new float [N];

	mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), 0, 0);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &mem);
	clSetKernelArg(kernel, 1, sizeof(int),    (void *) &N);

	clEnqueueNDRangeKernel( command_queue, kernel, 1, 0, &N, &N, 0, NULL, NULL);

	clEnqueueReadBuffer( command_queue, mem, CL_TRUE, 0, N * sizeof(float), pData, 0, 0, 0);

	clReleaseMemObject(mem);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete [] pDeviceId;
	delete [] pData;

	return 0;
}