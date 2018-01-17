/*
 * host.c
 *
 *  Created on: Jan 4, 2018
 *      Author: rchamberlain
 *
 *  Host code used darknet convolution kernel
 */
/*
	Simple host program for accessing openCL

*/

#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <cassert>
#include <time.h>
#include "activation_layer.h"
#include "convolutional_layer.h"
#include "darknet_kernels.h"

using namespace aocl_utils;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;

static size_t wgSize[3] = { 1, 1, 1 };
static size_t gSize[3] = { 1, 1, 1 };

// Convolution statics
static cl_command_queue queue_coeff = NULL;
static cl_command_queue queue_convolve = NULL;
static cl_kernel kernel_coeff_setup = NULL;
static cl_kernel kernel_convolve = NULL;
static cl_program program = NULL;
static cl_mem inbuf;
static cl_mem outbuf;
static cl_mem coeffbuf;
// Activation layer statics
static cl_command_queue queue_activation = NULL;
static cl_kernel kernel_activation = NULL;
static cl_mem inbuf_activation;
static cl_mem outbuf_activation;

static bool initiailised = false;

#define STRING_BUFFER_LEN 1024

static void display_device_info(cl_device_id device);


bool init() {
	cl_int status;

	if (!setCwdToExeDir()) {
		return false;
	}
	printf("Find platforms\n");
	// Get the OpenCL platform.
	platform = findPlatform("Intel");
	if (platform == NULL) {
		printf("ERROR: Unable to find Intel FPGA OpenCL platform.\n");
		return false;
	}
	printf("found platform\n");
	// User-visible output - Platform information
	{
		char char_buffer[STRING_BUFFER_LEN];
		printf("Querying platform for info:\n");
		printf("==========================\n");
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
	}

	// Query the available OpenCL devices.
	scoped_array<cl_device_id> devices;
	cl_uint num_devices=0;

	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	printf("num devices = %d\n",num_devices);

	// We'll just use the first device.
	device = devices[0];

	// Display some device information.
	display_device_info(device);

	// Create the context.
	context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the command queue.
	queue_coeff = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	queue_convolve = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	queue_activation = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile("./device/darknet", device);
	printf("Using AOCX File: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
	printf("Program created\n");

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");
	printf("Program built\n");


	const char *kernel_name = "basic_convolution_striped_load_coeffs_kernel_reduced_mem";// Kernel name, as defined in the CL file
	kernel_coeff_setup = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create basic_convolution_striped_load_coeffs_kernel kernel");

	const char *kernel_name2 = "basic_convolution_striped_kernel_reduced_mem";// Kernel name, as defined in the CL file
	kernel_convolve = clCreateKernel(program, kernel_name2, &status);
	checkError(status, "Failed to create basic_convolution_striped_kernel kernel");

	// Initialise activation layer
	const char *activation_kernel_name = "leaky_activate_fpga";// Kernel name, as defined in the CL file
	kernel_activation = clCreateKernel(program, activation_kernel_name, &status);
	checkError(status, "Failed to create leaky_activate_fpga kernel");

	printf("Finished initialisation\n");
	return true;
}

extern "C" void host_setup()
{
	init();

	cl_int status;

	inbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 1605632 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create mem buf inbuf");
	outbuf = clCreateBuffer(context, CL_MEM_READ_ONLY, 802816 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create mem buf outbuf");
	coeffbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4718592 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create mem buf coeffbuf");

	inbuf_activation = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024*1024 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create mem buf inbuf");
	outbuf_activation = clCreateBuffer(context, CL_MEM_READ_ONLY, 1024*1024 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create mem buf outbuf");

	initiailised = true;
}



extern "C" void forward_convolution_fpga(convolutional_layer l, network net)
{

   	if (!initiailised)
   	{
   		host_setup();
   	}
	int ln_rounded = l.n;
	if (l.n%STRIPES)
	{
		ln_rounded += STRIPES-(l.n%STRIPES);
	}

	float *new_weights = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*ln_rounded*l.size*l.size*sizeof(float));
	float *striped_input = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float));
	float *striped_output = (float*)malloc((ln_rounded)*l.out_w*l.out_w*sizeof(float));
	bool hardware_enabled = true;

	stripe_coefficients(l.c,l.n,l.size,l.weights,new_weights);
	stripe_input_data(l.c,l.w*l.w,net.input,striped_input);

	if (l.c < STRIPES)
		l.c = STRIPES;
	// RunOnHardware
	int y_div = 1;
	unsigned int InputBatchSize = (l.c*l.w*l.w)/STRIPES;
	while (InputBatchSize > MAX_INPUT_IMAGE_BATCH)
	{
		InputBatchSize = InputBatchSize >> 1;
		y_div = y_div*2;
	}

	if (hardware_enabled)
	{
		HardwareRunConvolution(striped_input,//net.input,
		   new_weights, striped_output,
		   l.batch,
		   l.groups,
		   l.nweights,
		   l.w,
		   l.out_w,
		   l.size,
		   l.pad,
		   l.c,
		   ln_rounded,
		   l.stride,
		   ((l.out_w*l.out_w)/MAX_BATCH_SIZE)/y_div,
		   y_div);
	}
	else
	{
		// Coefficient kernel ran on host
		basic_convolution_striped_load_coeffs_kernel_reduced_mem(
						   new_weights,
						   l.batch,
						   l.groups,
						   l.nweights,
						   l.w,
						   l.out_w,
						   l.size,
						   l.pad,
						   l.c,
						   ln_rounded,
						   l.stride,
						   ((l.out_w*l.out_w)/MAX_BATCH_SIZE)/y_div,
						   y_div
						   );
						   //input_batch);

		basic_convolution_striped_kernel_reduced_mem(striped_input,//net.input,
						   new_weights, striped_output,
						   l.batch,
						   l.groups,
						   l.nweights,
						   l.w,
						   l.out_w,
						   l.size,
						   l.pad,
						   l.c,
						   ln_rounded,
						   l.stride,
						  ((l.out_w*l.out_w)/MAX_BATCH_SIZE)/y_div,
						  y_div);
						  //input_batch);// Input subdivisions
	}

	remove_stripes(l.n,(l.out_w*l.out_w),striped_output,l.output);
	clear_channel();
	free(new_weights);
	free(striped_input);
	free(striped_output);

}



extern "C" void activate_array(float* x, const int n, const ACTIVATION a)
{
	// data
	cl_int status;
   	int i;
   	unsigned int work_group_size = 1;
   	if (!initiailised)
   	{
   		host_setup();
   	}
	// code
	if (a == LEAKY)
	{
		// Call the FPGA kernel
		fprintf(stdout, "\n\n*** [FPGA] Launching the kernel...\n\n");
  		// Configure work set over which the kernel will execute
  		size_t wgSize[3] = {work_group_size, 1, 1};
  		size_t gSize[3] = {work_group_size, 1, 1};
  		// Create input variables to pass input-output data with the kernel
		// Write input to the kernel
		clEnqueueWriteBuffer(queue_activation, inbuf_activation, CL_TRUE, 0, sizeof(float) * n, x, 0, NULL, NULL);
		//Set arguements
		clSetKernelArg(kernel_activation, 0, sizeof(cl_mem), &inbuf_activation);
		clSetKernelArg(kernel_activation, 1, sizeof(cl_mem), &outbuf_activation);
		clSetKernelArg(kernel_activation, 2, sizeof(int), (void*)&n);
		// Launch the kernel
  		status = clEnqueueNDRangeKernel(queue_activation, kernel_activation, 1, NULL, gSize, wgSize, 0, NULL, NULL);
  		checkError(status, "Failed to launch kernel");
		// Reading output
		clEnqueueReadBuffer(queue_activation, outbuf_activation, CL_TRUE, 0, sizeof(float) * n, x, 0, NULL, NULL);
		// Completed
		status = clFinish(queue_activation);
		checkError(status, "Failed to finish");
		fprintf(stdout, "\n\n*** [FPGA] Kernel execution completed\n\n");
	}
	else
	{
		// perform the activation using host code
    		for(i = 0; i < n; ++i)
		{
        		x[i] = activate(x[i], a);
    		}
	}
}



extern "C" void HardwareRunConvolution(float *input, float *coeffs, float *output,
		int batch,
		int groups,
		int nweights,
		int size,
		int out_size,
		int kernel_size,
		int pad,
		int in_f,
		int out_f,
		int stride,
		int batches_of_49,
		int y_div)
{
	cl_int status;
	// Write input to kernel global memory
	clEnqueueWriteBuffer(queue_coeff, coeffbuf, CL_TRUE, 0, sizeof(float) * kernel_size*kernel_size*in_f*out_f, coeffs, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue_convolve, inbuf, CL_TRUE, 0, sizeof(float) * size*size*in_f, input, 0, NULL, NULL);

	int i,j;
	//Set arguements
	clSetKernelArg(kernel_coeff_setup, 0, sizeof(cl_mem), &coeffbuf);
	clSetKernelArg(kernel_coeff_setup, 1, sizeof(int), &batch);
	clSetKernelArg(kernel_coeff_setup, 2, sizeof(int), &groups);
	clSetKernelArg(kernel_coeff_setup, 3, sizeof(int), &nweights);
	clSetKernelArg(kernel_coeff_setup, 4, sizeof(int), &size);
	clSetKernelArg(kernel_coeff_setup, 5, sizeof(int), &out_size);
	clSetKernelArg(kernel_coeff_setup, 6, sizeof(int), &kernel_size);
	clSetKernelArg(kernel_coeff_setup, 7, sizeof(int), &pad);
	clSetKernelArg(kernel_coeff_setup, 8, sizeof(int), &in_f);
	clSetKernelArg(kernel_coeff_setup, 9, sizeof(int), &out_f);
	clSetKernelArg(kernel_coeff_setup, 10, sizeof(int), &stride);
	clSetKernelArg(kernel_coeff_setup, 11, sizeof(int), &batches_of_49);
	clSetKernelArg(kernel_coeff_setup, 12, sizeof(int), &y_div);


	clSetKernelArg(kernel_convolve, 0, sizeof(cl_mem), &inbuf);
	clSetKernelArg(kernel_convolve, 1, sizeof(cl_mem), &coeffbuf);
	clSetKernelArg(kernel_convolve, 2, sizeof(cl_mem), &outbuf);
	clSetKernelArg(kernel_convolve, 3, sizeof(int), &batch);
	clSetKernelArg(kernel_convolve, 4, sizeof(int), &groups);
	clSetKernelArg(kernel_convolve, 5, sizeof(int), &nweights);
	clSetKernelArg(kernel_convolve, 6, sizeof(int), &size);
	clSetKernelArg(kernel_convolve, 7, sizeof(int), &out_size);
	clSetKernelArg(kernel_convolve, 8, sizeof(int), &kernel_size);
	clSetKernelArg(kernel_convolve, 9, sizeof(int), &pad);
	clSetKernelArg(kernel_convolve, 10, sizeof(int), &in_f);
	clSetKernelArg(kernel_convolve, 11, sizeof(int), &out_f);
	clSetKernelArg(kernel_convolve, 12, sizeof(int), &stride);
	clSetKernelArg(kernel_convolve, 13, sizeof(int), &batches_of_49);
	clSetKernelArg(kernel_convolve, 14, sizeof(int), &y_div);


	printf("launch kernel\n");
	status = clEnqueueNDRangeKernel(queue_coeff, kernel_coeff_setup, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");

	status = clEnqueueNDRangeKernel(queue_convolve, kernel_convolve, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");

	printf("kernels launched\n");
	status = clFinish(queue_convolve);
	printf("kernels finished\n");

	clEnqueueReadBuffer(queue_convolve, outbuf, CL_TRUE, 0, sizeof(float) * out_size*out_size*out_f, output, 0, NULL, NULL);
	// Check queue has completed, in case we are not using a blocking function.
	checkError(status, "Failed to finish");

	return;
}


// Free the resources allocated during initialization
void cleanup() {
	if (kernel_coeff_setup) {
		clReleaseKernel(kernel_coeff_setup);
	}
	if (kernel_convolve) {
		clReleaseKernel(kernel_convolve);
	}
	if (kernel_activation) {
		clReleaseKernel(kernel_convolve);
	}

	if (program) {
		clReleaseProgram(program);
	}
	if (queue_coeff) {
		clReleaseCommandQueue(queue_coeff);
	}
	if (queue_convolve) {
		clReleaseCommandQueue(queue_convolve);
	}
	if (queue_convolve) {
		clReleaseCommandQueue(queue_activation);
	}
	if (context) {
		clReleaseContext(context);
	}
}


// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name) {
	cl_ulong a;
	clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
	printf("%-40s = %lu\n", name, a);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name) {
	cl_uint a;
	clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
	printf("%-40s = %u\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name) {
	cl_bool a;
	clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
	printf("%-40s = %s\n", name, (a ? "true" : "false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char* name) {
	char a[STRING_BUFFER_LEN];
	clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
	printf("%-40s = %s\n", name, a);
}
// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device) {

	printf("Querying device for info:\n");
	printf("========================\n");
	device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
	device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
	device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
	device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
	device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
	device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
	device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
	device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
	device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
	device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
	device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
	device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
	device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
	device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

	{
		cl_command_queue_properties ccp;
		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
		printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
		printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
	}
}

