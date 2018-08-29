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
static cl_kernel kernel_pool = NULL;
static cl_kernel kernel_shortcut = NULL;
static cl_kernel kernel_yolo = NULL;
static cl_kernel kernel_upsample = NULL;
static cl_kernel kernel_route = NULL;

static cl_program program = NULL;
static cl_mem inbuf;
static cl_mem outbuf;
static cl_mem coeffbuf;
static cl_mem scalebuf;
// Activation layer statics
static cl_command_queue queue_activation = NULL;
static cl_kernel kernel_activation = NULL;
static cl_mem div_sqrt_variance_buf;
static cl_mem rolling_mean_buf;
static cl_mem bias_scales_buf;
static cl_mem biases_buf;

static bool initiailised = false;

#define STRING_BUFFER_LEN 1024

static void display_device_info(cl_device_id device);


bool init(bool binary) {
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

	std::string binary_file;
	if (binary)
		binary_file= getBoardBinaryFile("./kernels/bin256", device);
	else
		binary_file= getBoardBinaryFile("./device/darknet", device);
	printf("Using AOCX File: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
	printf("Program created\n");

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");
	printf("Program built\n");


	if (binary)
	{
		const char *kernel_name = "conv_coeffs_binary_subblock_fpga";// Kernel name, as defined in the CL file
		kernel_coeff_setup = clCreateKernel(program, kernel_name, &status);
		checkError(status, "Failed to create basic_convolution_striped_load_coeffs_kernel_binary_bits kernel");

		const char *kernel_name2 = "conv_binary_subblock_fpga";// Kernel name, as defined in the CL file
		kernel_convolve = clCreateKernel(program, kernel_name2, &status);
		checkError(status, "Failed to create basic_convolution_striped_kernel_binary kernel");
	}
	else
	{
		const char *kernel_name = "basic_convolution_striped_load_coeffs_kernel_reduced_mem";// Kernel name, as defined in the CL file
		kernel_coeff_setup = clCreateKernel(program, kernel_name, &status);
		checkError(status, "Failed to create basic_convolution_striped_load_coeffs_kernel kernel");

		const char *kernel_name2 = "basic_convolution_striped_kernel_reduced_mem";// Kernel name, as defined in the CL file
		kernel_convolve = clCreateKernel(program, kernel_name2, &status);
		checkError(status, "Failed to create basic_convolution_striped_kernel kernel");
	}


	// Initialise activation layer
	const char *activation_kernel_name = "leaky_activate_fpga";// Kernel name, as defined in the CL file
	kernel_activation = clCreateKernel(program, activation_kernel_name, &status);
	checkError(status, "Failed to create leaky_activate_fpga kernel");

	const char *pool_kernel_name = "hw_maxpooling_fpga_x2_striped";// Kernel name, as defined in the CL file
	kernel_pool = clCreateKernel(program, pool_kernel_name, &status);
	checkError(status, "Failed to create hw_maxpooling_fpga_x2_striped");

	const char *shortcut_kernel_name = "shortcut_layer_fpga";// Kernel name, as defined in the CL file
	kernel_shortcut = clCreateKernel(program, shortcut_kernel_name, &status);
	checkError(status, "Failed to create shortcut_layer_fpga kernel");

	const char *upsample_x2_kernel_name = "upsample_x2_fpga";// Kernel name, as defined in the CL file
	kernel_upsample = clCreateKernel(program, upsample_x2_kernel_name, &status);
	checkError(status, "Failed to create upsample_x2_fpga kernel");

	const char *yolo_kernel_name = "yolo_layer_fpga";// Kernel name, as defined in the CL file
	kernel_yolo = clCreateKernel(program, yolo_kernel_name, &status);
	checkError(status, "Failed to create yolo_layer_fpga kernel");

	const char *route_kernel_name = "route_layer_fpga";// Kernel name, as defined in the CL file
	kernel_route = clCreateKernel(program, route_kernel_name, &status);
	checkError(status, "Failed to create route_layer_fpga kernel");


	printf("Finished initialisation\n");
	return true;
}

extern "C" void host_setup(int binary)
{
	init(binary?true:false);
	// Setup common buffer of input and output
	cl_int status;
	inbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 512*512*32 * sizeof(float), NULL, &status);
	outbuf  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512*512*64 * sizeof(float), NULL, &status);
	coeffbuf  = clCreateBuffer(context, CL_MEM_READ_ONLY, 1024*1024*9 * sizeof(float), NULL, &status);
	scalebuf  = clCreateBuffer(context, CL_MEM_READ_ONLY, 1024 * sizeof(float), NULL, &status);


	initiailised = true;
}

extern "C" cl_context GetFPGAContext()
{
	return context;
}


extern "C" void setup_layer_fpga_opencl_interfaces(convolutional_layer l)
{

}

float total_kernel_time = 0;

extern "C" void forward_convolution_fpga(convolutional_layer l, network net)
{
   	int ln_rounded = l.n;
	if (l.n%STRIPES)
	{
		ln_rounded += STRIPES-(l.n%STRIPES);
	}
 //printf("l.binary = %s\n",l.binary?"TRUE":"FALSE");

#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif
	float *new_weights=NULL;
	float *striped_input=NULL;
	float *striped_output=NULL;
	//if (!hardware_enabled || (l.first == 1))
	{
		new_weights = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*ln_rounded*l.size*l.size*sizeof(float)*2);
		stripe_coefficients(l.c,l.n,l.size,l.weights,new_weights);
	}
	//if (!hardware_enabled || (l.fpga_load ==1))
	{
		striped_input = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);
		stripe_input_data(l.c,l.w*l.w,net.input,striped_input);
		total_kernel_time = 0.0f;
	}
	//if (!hardware_enabled || (l.fpga_save == 1))
	{
		striped_output = (float*)alignedMalloc((ln_rounded)*l.out_w*l.out_w*sizeof(float)*2);
	}
	int input_features = l.c;
	if (input_features < STRIPES)
		input_features = STRIPES;
	// RunOnHardware
	int y_div = 1;
	// Increase the input batch size if striding


	float InputBatchSize = (((float)(input_features*l.w*l.w))/STRIPES);
	printf("InputBatchSize = %f\n",InputBatchSize);
	while (InputBatchSize > (float)(MAX_INPUT_IMAGE_BATCH))
	{
		InputBatchSize = InputBatchSize / 2.0f;
		printf("InputBatchSize = %f\n",InputBatchSize);
		y_div = y_div*2;
	}
	/*int out_size_y_inc;
	int in_size_y_inc;
	out_size_y_inc = l.out_w/y_div;
	in_size_y_inc = out_size_y_inc*l.stride;
	y_div = l.w/in_size_y_inc;
	y_div += (l.w%in_size_y_inc)?1:0;*/
	printf("y_div = %d\n",y_div);


	unsigned int chunk_size = MAX_BATCH_SIZE; // number of pixels


	float batches_of_169 = InputBatchSize/(MAX_BATCH_SIZE*l.stride*l.stride);
	batches_of_169 = ((l.out_w/y_div)*l.out_w)/MAX_BATCH_SIZE;
	int no_chunks = 1;//
	printf("Batches of 169 = %f\n", batches_of_169);
	printf(" inputs = %d, buffered_size = %d\n",l.w*l.w,y_div*MAX_BATCH_SIZE);
	//printf("in = %d:out = %d\n",l.fpga_load,l.fpga_save);
	if (hardware_enabled)
	{
		HardwareRunConvolution(striped_input,//net.input,
		   new_weights, striped_output,
		   ln_rounded,
		   l);

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
						   input_features,
						   ln_rounded,
						   l.stride,
						   batches_of_169,
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
						   batches_of_169,
						   y_div);
						  //input_batch);// Input subdivisions


		float div_sqrt_variance[2048];
		if (l.batch_normalize)
		for (int i = 0; i < l.out_c; i++)
		{
			div_sqrt_variance[i] = 1.0f/(sqrt(l.rolling_variance[i]) + .000001f);
		}
		float activation_data[1024*4];
		printf("l.out_c = %d\n",l.out_c);
		for (int i = 0; i < l.out_c; i++) activation_data[i] = div_sqrt_variance[i];
		for (int i = 0; i < l.out_c; i++) activation_data[i+l.out_c] = l.rolling_mean[i];
		for (int i = 0; i < l.out_c; i++) activation_data[i+l.out_c*2] = l.scales[i];
		for (int i = 0; i < l.out_c; i++) activation_data[i+l.out_c*3] = l.biases[i];


		leaky_activate_fpga (striped_output, striped_output, l.batch,l.out_c,l.out_h*l.out_w,
				activation_data,//div_sqrt_variance,
								//  l.rolling_mean,
								//  l.scales,
								//  l.biases,
							l.batch_normalize?1:0,
							FPGA_LEAKY);
	}
	//printf("clear data\n");
	//if (!hardware_enabled || (l.fpga_save ==1))
	{
		remove_stripes(ln_rounded,(l.out_w*l.out_w),striped_output,l.output);
		alignedFree(striped_output);
	}
	//if (!hardware_enabled || (l.fpga_load ==1))
	{
		//clear_channel();
		alignedFree(striped_input);
	}
	//if (!hardware_enabled || (l.first == 1)) // First pass
		alignedFree(new_weights);

}

// Create coefficients striped for binary networks
// Current limitation is 32 stripes, but will change this once validated
// Binary weights need a scaling factor to be applied to final results
inline void SetBit(unsigned int *Bits,unsigned int pos,unsigned int val)
{
	*Bits |= (val<<pos);
}

extern "C" void stripe_coefficients_binary(int in_f,int out_f, int kernel_size,float *in,unsigned int *out,float *scale)
{
	int size = (kernel_size*kernel_size);
	int out_index=0;
	int i,j,k,q,p;

	int scale_count = 0;
	for (i = 0; i < out_f; i+= STRIPES)
	{
		for ( q = 0; q < STRIPES; q++)
		{
			float val =in[(q+i)*in_f*size];
			if (val < 0) val=-val;
			scale[scale_count++] = val/BINARY_FLOAT_SCALE;
		}
		for ( j = 0; j < in_f; j+= STRIPES)
		{


			for ( k = 0; k < size;k++)
			{
				for ( q = 0; q < STRIPES; q++)
				{
					unsigned int bits;
					bits = 0;
					for ( p = 0; p < STRIPES; p++)
					if (p < in_f)
					{
						int index_in = (j+p)*size; // Offset in input
							index_in += (q+i)*in_f*size; // Offset in output
							index_in += k; // Offset in kernel
						float val =in[index_in];
						//bits = (bits<<1)|bit;
						int bit=val<0?0:1;
						SetBit(&bits,p,bit);
					}
					out[out_index++] = bits;
				}
			}
		}
	}
}

#ifdef OLD
extern "C" void forward_convolution_fpga_binary(convolutional_layer l, network net)
{
	printf("forward_convolution_fpga_binary\n");
	int ln_rounded = l.n;
	if (l.n%STRIPES)
	{
		ln_rounded += STRIPES-(l.n%STRIPES);
	}

	float *new_weights = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*ln_rounded*l.size*l.size*sizeof(float)*2);
	unsigned int *new_weights_binary = (unsigned int*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*ln_rounded*l.size*l.size*sizeof(unsigned int)*2);
	float *striped_input = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);
	float *striped_output = (float*)alignedMalloc((ln_rounded)*l.out_w*l.out_w*sizeof(float)*2);
	float *binary_scale = (float*)alignedMalloc(l.n*sizeof(float)*64);

#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif

	stripe_coefficients_binary(l.c,l.n,l.size,l.weights,new_weights_binary,binary_scale);
	stripe_coefficients(l.c,l.n,l.size,l.weights,new_weights);
	stripe_input_data(l.c,l.w*l.w,net.input,striped_input);
	int l_c_orig = l.c;
	if (l.c < STRIPES)
		l.c = STRIPES;
	// RunOnHardware
	int y_div = 1;
	unsigned int InputBatchSize = (l.c*l.w*l.w)/STRIPES;
	while (InputBatchSize > MAX_INPUT_IMAGE_BATCH)
	{
		InputBatchSize = InputBatchSize / 2.0f;
		y_div = y_div*2;
	}
	printf("y_div = %d\n",y_div);
	unsigned int chunk_size = MAX_BATCH_SIZE; // number of pixels
	float batches_of_169 = InputBatchSize/(MAX_BATCH_SIZE*l.stride*l.stride);
	batches_of_169 = ((l.out_w/y_div)*l.out_w)/MAX_BATCH_SIZE;
	int no_chunks = 1;//
	printf("Batches of 169 = %f\n", batches_of_169);
	printf(" inputs = %d, buffered_size = %d\n",l.w*l.w,y_div*MAX_BATCH_SIZE);

	if (hardware_enabled)
	{
		HardwareRunConvolutionBinary(
		  striped_input,//net.input,
		   new_weights_binary, striped_output, binary_scale,
		   l.batch,
		   l.groups,
		   l.nweights,
		   l.w,
		   l.out_w,
		   l.size,
		   l.pad,
		   l.c,
		   l_c_orig,
		   ln_rounded,
		   l.stride,
		   ((l.out_w*l.out_w)/MAX_BATCH_SIZE)/y_div,
		   y_div);
	}
	else
	{
		// Coefficient kernel ran on host
		/*basic_convolution_striped_load_coeffs_kernel_binary(
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
						   //input_batch);*/

		basic_convolution_striped_load_coeffs_kernel_binary_bits(
						   new_weights_binary,
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

		basic_convolution_striped_kernel_binary(striped_input,//net.input,
						   binary_scale, striped_output,
						   l.batch,
						   l.groups,
						   l.nweights,
						   l.w,
						   l.out_w,
						   l.w/y_div,
						   l.out_w/y_div,
						   l.size,
						   l.pad,
						   l.c,
						   l_c_orig,
						   ln_rounded,
						   l.stride,
						  ((l.out_w*l.out_w)/MAX_BATCH_SIZE)/y_div,
						  y_div);
						  //input_batch);// Input subdivisions
	}

	remove_stripes(l.n,(l.out_w*l.out_w),striped_output,l.output);
	//clear_channel();
	alignedFree(new_weights);
	alignedFree(striped_input);
	alignedFree(striped_output);
	alignedFree(binary_scale);

}
#endif


extern "C" void forward_convolution_fpga_binary_v2(convolutional_layer l, network net)
{
   	int ln_rounded = l.n;
	if (l.n%STRIPES)
	{
		ln_rounded += STRIPES-(l.n%STRIPES);
	}
#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif
	float *new_weights=NULL;
	float *striped_input=NULL;
	float *striped_output=NULL;
	unsigned int *new_weights_binary=NULL;
	float *binary_scale=NULL;


	if (!hardware_enabled || (l.first == 1))
	{
		new_weights_binary = (unsigned int*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*ln_rounded*l.size*l.size*sizeof(unsigned int)*2);
		binary_scale = (float*)alignedMalloc(32*ln_rounded*sizeof(float));
		new_weights = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*ln_rounded*l.size*l.size*sizeof(float)*2);
		stripe_coefficients_binary(l.c,l.n,l.size,l.weights,new_weights_binary,binary_scale);

		stripe_coefficients(l.c,l.n,l.size,l.weights,new_weights);
	}
	if (!hardware_enabled || (l.fpga_load == 1))
	{
		striped_input = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);
		stripe_input_data(l.c,l.w*l.w,net.input,striped_input);
		total_kernel_time = 0.0f;
	}
	if (!hardware_enabled || (l.fpga_save == 1))
	{
		striped_output = (float*)alignedMalloc((ln_rounded)*l.out_w*l.out_w*sizeof(float)*2);
		printf("Striped output size = %d\n",l.out_w*l.out_w*ln_rounded);
	}
	int input_features = l.c;
	if (input_features < STRIPES)
		input_features = STRIPES;
	// RunOnHardware
	// Increase the input batch size if striding

	// Calculate the maximum sub-block size that fits in the available M20K memory reserved
	// Number of feature groups required
	int f_groups = input_features/STRIPES;
	int InputBatchSize = f_groups*(l.w + l.size -1)*(l.w + l.size -1);
	int divx = 1,divy=1;

	while (InputBatchSize > (float)(MAX_INPUT_IMAGE_BATCH))
	{
		if (((l.w&0x1)==0))
		if (((l.w/(divx*2))&0x1) == 0) // Cannot divide non power of two in x
		{
			divx *= 2;
			InputBatchSize = f_groups*((l.w/divx) + l.size -1)*((l.w/divy) + l.size -1);
		}
		if (InputBatchSize > (float)(MAX_INPUT_IMAGE_BATCH))
		{
			divy *= 2;
			InputBatchSize = f_groups*((l.w/divx) + l.size -1)*((l.w/divy) + l.size -1);
		}

	}

	// Make sub block size 416/4
	int sub_block_size_x = l.w/divx;
	int sub_block_size_y = l.w/divy;

	// divy can not be odd if striding.
	if ((sub_block_size_y&0x1) && (l.stride != 1))
	{
		sub_block_size_y &= 0xfffe;

	}
	divy = l.w/sub_block_size_y;
	if (l.w%sub_block_size_y)
		divy++;
	int no_sub_blocks = divx*divy;

	// Block size needs to be bigger than the number of coeffs to load.
	if ((sub_block_size_x*sub_block_size_y/(l.stride*l.stride)) < 8)
		assert("Sub block size to small!!!");

	if (hardware_enabled)
	{
		HardwareRunConvolutionBinary(striped_input,//net.input,
		   new_weights_binary, striped_output,binary_scale,
		   ln_rounded,
		   l,
		   sub_block_size_x,
		   sub_block_size_y,
		   divx,
		   divy,
		   net);


	}
	else
	{
		// while




		printf("divx = %d\n",divx);
		printf("divy = %d\n",divy);
		printf("sub_block_size_x = %d\n",sub_block_size_x);
		printf("sub_block_size_y = %d\n",sub_block_size_y);
		printf("stride = %d\n",l.stride);

		conv_coeffs_binary_subblock_fpga(
						   new_weights_binary,//new_weights,
						   l.batch,
						   l.groups,
						   l.nweights,
						   l.w,
						   l.out_w,
						   l.size,
						   l.pad,
						   input_features,
						   ln_rounded,
						   l.stride,
						   1,//batches_of_169,
						   no_sub_blocks,
						   (input_features*l.size*l.size)
						   );




#ifdef V3
		float div_sqrt_variance[2048];

		if (l.batch_normalize)
		for (int i = 0; i < l.out_c; i++)
		{
			div_sqrt_variance[i] = 1.0f/(sqrt(l.rolling_variance[i]) + .000001f);
		}
		float activation_data[1024*4];
		for (int i = 0; i < l.out_c; i++) activation_data[i] = div_sqrt_variance[i];
		if ( l.batch_normalize)
		{
			for (int i = 0; i < l.out_c; i++) activation_data[i+ln_rounded] = l.rolling_mean[i];
			for (int i = 0; i < l.out_c; i++) activation_data[i+ln_rounded*2] = l.scales[i];
		}
		for (int i = 0; i < l.out_c; i++) activation_data[i+ln_rounded*3] = l.biases[i];


		//leaky_activate_fpga (striped_output, striped_output, l.batch,l.out_c,l.out_h*l.out_w,
		//		activation_data,//div_sqrt_variance,
		//						//  l.rolling_mean,
		//						//  l.scales,
		//						//  l.biases,
		//						  l.batch_normalize?1:0,
		//						((l.activation == LEAKY)||(l.activation == BINARY))? FPGA_LEAKY:FPGA_LINEAR);
		int activation = ((l.activation == LEAKY)|| (l.activation == BINARY))?FPGA_LEAKY:FPGA_LINEAR;

		conv_binary_subblock_fpga_v3(striped_input,//net.input,
						   new_weights_binary, striped_output,

						   divy,
						   divx,
						   sub_block_size_x,
						   sub_block_size_y,

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
						   0, // Not used
						   0,
						   binary_scale,
						   0,//batch_size, not used
						   0,
						   0,
						   l.c,
						   activation_data,
						   l.batch_normalize,
						   activation
						   	 );

#endif

#ifdef V4

		float div_sqrt_variance[2048];

		if (l.batch_normalize)
		for (int i = 0; i < l.out_c; i++)
		{
			div_sqrt_variance[i] = 1.0f/(sqrt(l.rolling_variance[i]) + .000001f);
		}
		float activation_data[1024*4];
		for (int i = 0; i < l.out_c; i++) activation_data[(i<<2)] = div_sqrt_variance[i];
		if ( l.batch_normalize)
		{
			for (int i = 0; i < l.out_c; i++) activation_data[(i<<2)+1] = l.rolling_mean[i];
			for (int i = 0; i < l.out_c; i++) activation_data[(i<<2)+2] = l.scales[i];
		}
		for (int i = 0; i < l.out_c; i++) activation_data[(i<<2)+3] = l.biases[i];


		//leaky_activate_fpga (striped_output, striped_output, l.batch,l.out_c,l.out_h*l.out_w,
		//		activation_data,//div_sqrt_variance,
		//						//  l.rolling_mean,
		//						//  l.scales,
		//						//  l.biases,
		//						  l.batch_normalize?1:0,
		//						((l.activation == LEAKY)||(l.activation == BINARY))? FPGA_LEAKY:FPGA_LINEAR);
		int activation = ((l.activation == LEAKY)|| (l.activation == BINARY))?FPGA_LEAKY:FPGA_LINEAR;

		conv_binary_subblock_fpga_v4(striped_input,//net.input,
						   new_weights_binary, striped_output,

						   divy,
						   divx,
						   sub_block_size_x,
						   sub_block_size_y,

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
						   0, // Not used
						   0,
						   binary_scale,
						   0,//batch_size, not used
						   0,
						   0,
						   l.c,
						   10
						   	 );


		conv_activations_v4(striped_output,//net.input,
						   divy,
						   divx,
						   sub_block_size_x,
						   sub_block_size_y,

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
						   0, // Not used
						   0,
						   binary_scale,
						   0,//batch_size, not used
						   0,
						   0,
						   l.c,
						   activation_data,
						   l.batch_normalize,
						   activation
						   	 );
#endif
						  //input_batch);// Input subdivisions
		/*leaky_activate_fpga (striped_output, striped_output, l.batch,l.out_c,l.out_h*l.out_w,
				activation_data,//div_sqrt_variance,
								//  l.rolling_mean,
								//  l.scales,
								//  l.biases,
								  l.batch_normalize?1:0,
								((l.activation == LEAKY)||(l.activation == BINARY))? FPGA_LEAKY:FPGA_LINEAR);*/


	}
	
	if (!hardware_enabled || (l.fpga_save ==1))
	{
		// Save results
		remove_stripes(ln_rounded,(l.out_w*l.out_w),striped_output,l.output);
		alignedFree(striped_output);
	}
	if (!hardware_enabled || (l.fpga_load ==1))
	{
		//clear_channel();
		alignedFree(striped_input);
	}
	//
	if (!hardware_enabled || (l.first == 1)) // First pass
	{
		alignedFree(new_weights);
		alignedFree(new_weights_binary);
		alignedFree(binary_scale);
	}

}




extern "C" void forward_upsample_layer_fpga(const layer l, network net)
{
#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif
	if (hardware_enabled)
	{

	cl_mem in_mem = net.layers[l.id-1].fpga_outbuf;
	cl_mem out_mem = l.fpga_outbuf;
	int insize = l.w*l.w;
	int size = l.out_w*l.out_h;
	printf("FPGA upsample layer!\n");


	clSetKernelArg(kernel_upsample, 0, sizeof(cl_mem), &in_mem);
	clSetKernelArg(kernel_upsample, 1, sizeof(cl_mem), &out_mem);
	clSetKernelArg(kernel_upsample, 2, sizeof(int), &l.batch);
	clSetKernelArg(kernel_upsample, 3, sizeof(int), &l.c);
	clSetKernelArg(kernel_upsample, 4, sizeof(int), &size);
	clSetKernelArg(kernel_upsample, 5, sizeof(unsigned short), &l.out_w);
	clSetKernelArg(kernel_upsample, 6, sizeof(unsigned short), &l.out_h);
	clSetKernelArg(kernel_upsample, 7, sizeof(int), &insize);
	clSetKernelArg(kernel_upsample, 8, sizeof(int), &size);
	
	cl_event kernel_event;
	cl_int status;
	status = clEnqueueNDRangeKernel(queue_coeff, kernel_upsample, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel");
	status = clFinish(queue_coeff);

	if (!hardware_enabled || (l.fpga_save ==1))
	{
		float *striped_output = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*l.out_w*l.out_w*sizeof(float));
		clEnqueueReadBuffer(queue_coeff, out_mem, CL_TRUE, 0, sizeof(float) *size * l.c , striped_output, 0, NULL, NULL);
		remove_stripes((l.c < STRIPES?STRIPES:l.c),(l.out_w*l.out_w),striped_output,l.output);
	}

	}
	else
	{
		float *striped_input = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float));
		float *striped_output = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*l.out_w*l.out_w*sizeof(float));
		stripe_input_data(l.c,l.w*l.w,net.input,striped_input);

		upsample_x2_fpga( striped_input, striped_output,
				      l.batch,
					  l.c,
					  l.out_w*l.out_h,
					  l.out_w,
					  l.out_h,
					  l.w*l.w,
					  l.out_w*l.out_h);

		remove_stripes((l.c < STRIPES?STRIPES:l.c),(l.out_w*l.out_w),striped_output,l.output);


		free(striped_input);
		free(striped_output);
	}
}


extern "C" void forward_shortcut_layer_fpga(const route_layer l, network net)
{
#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif
	printf("fpga short cut\n");
	{
		float *striped_input=NULL;
		float *striped_input_add=NULL;
		float *striped_output=NULL;
		float *unstriped_fpga=NULL;
		int ln_rounded = l.out_c;
		if (l.out_c%STRIPES)
		{
			ln_rounded += STRIPES-(l.out_c%STRIPES);
		}
		int i;
		if (!hardware_enabled || l.fpga_load)
		{
			printf("upload short cut data\n");
			striped_input_add = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);
			striped_input = (float*)malloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);
		}
		if (!hardware_enabled || l.fpga_save)
		{
			striped_output = (float*)alignedMalloc((ln_rounded)*l.out_w*l.out_w*sizeof(float)*2);
		}
		if (!hardware_enabled || l.fpga_load == 1)
		{
			stripe_input_data(l.c,l.w*l.w, net.layers[l.index].output,striped_input_add);
			stripe_input_data(l.c,l.w*l.w, net.input,striped_input);
			//clEnqueueWriteBuffer(queue_coeff, inbuf, CL_TRUE, 0, sizeof(float) * (l.c < STRIPES?STRIPES:l.c)*l.w*l.w, striped_input, 0, NULL, NULL);			
		}



		if (hardware_enabled)
		{
			//if (!hardware_enabled || l.fpga_load == 1)
			//if (!hardware_enabled || l.fpga_load == 1)

			int input_block_size = l.batch*l.w*l.h*l.c;
			int output_size = l.out_h*l.out_w;
			cl_mem in_mem = net.layers[l.id-1].fpga_outbuf;// Use previous layer output buffer.
			cl_mem mem = net.layers[l.index].fpga_outbuf;
			printf("mem = %x\n",mem);
			printf("l.fpga_outbuf = %x\n",l.fpga_outbuf);
			// Get buffer of previous generated convolution output
			clSetKernelArg(kernel_shortcut, 0, sizeof(int), &input_block_size);
			clSetKernelArg(kernel_shortcut, 1, sizeof(int), &l.batch);
			clSetKernelArg(kernel_shortcut, 2, sizeof(int), &l.w);
			clSetKernelArg(kernel_shortcut, 3, sizeof(int), &l.h);
			clSetKernelArg(kernel_shortcut, 4, sizeof(int), &l.c);
			clSetKernelArg(kernel_shortcut, 5, sizeof(cl_mem), &mem); // Points to layer to append
			clSetKernelArg(kernel_shortcut, 6, sizeof(int), &l.out_w);
			clSetKernelArg(kernel_shortcut, 7, sizeof(int), &l.out_h);
			clSetKernelArg(kernel_shortcut, 8, sizeof(int), &l.out_c);
			clSetKernelArg(kernel_shortcut, 9, sizeof(float), &l.alpha);
			clSetKernelArg(kernel_shortcut, 10, sizeof(float), &l.beta);
			clSetKernelArg(kernel_shortcut, 11, sizeof(cl_mem), &in_mem);
			clSetKernelArg(kernel_shortcut, 12, sizeof(cl_mem), &l.fpga_outbuf);
			clSetKernelArg(kernel_shortcut, 13, sizeof(cl_mem), &l.fpga_outbuf);



			cl_event kernel_event;
			cl_int status;
			//printf("Max pool launch kernel\n");
			status = clEnqueueNDRangeKernel(queue_coeff, kernel_shortcut, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
			checkError(status, "Failed to launch kernel");
			status = clFinish(queue_coeff);
			if (!hardware_enabled || l.fpga_save == 1)
				clEnqueueReadBuffer(queue_coeff, l.fpga_outbuf, CL_TRUE, 0, sizeof(float) * (l.c < STRIPES?STRIPES:l.c)*l.out_w*l.out_w, striped_output, 0, NULL, NULL);

		}
		else
		{
			shortcut_layer_fpga(l.batch*l.w*l.h*l.c, l.batch, l.w, l.h, l.c, striped_input_add, l.out_w, l.out_h, l.out_c,  l.alpha, l.beta, striped_input,striped_output, striped_output);
		}
		if (!hardware_enabled || l.fpga_save == 1)
			remove_stripes(ln_rounded,(l.out_w*l.out_w),striped_output,l.output);

		// Linear activation, do nothing!
		if (!(l.activation == LINEAR))
		{
			printf("Warning non linear activation not supported: %s:%d\n",__FILE__,__LINE__);
			exit(1);
		}
		if (!hardware_enabled || l.fpga_save == 1)
		{
			alignedFree(striped_input);	alignedFree(striped_input_add);
			alignedFree(striped_output);
		}
	}
	printf("end shortcut layer\n");
}


extern "C" void forward_maxpool_layer_fpga(const maxpool_layer l, network net)
{

#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif
    // Replace with FPGA optimised version
	float *striped_input;
	float *striped_output;
	if (!hardware_enabled || l.fpga_load)
		striped_input = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);
	if (!hardware_enabled || l.fpga_save)
		striped_output = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*l.out_w*l.out_w*sizeof(float)*2);

	if (!hardware_enabled || l.fpga_load == 1)
		stripe_input_data(l.c,l.w*l.w,net.input,striped_input);


	if (hardware_enabled)
	{
		if (l.fpga_load == 1)
			clEnqueueWriteBuffer(queue_coeff, inbuf, CL_TRUE, 0, sizeof(float) * (l.c < STRIPES?STRIPES:l.c)*l.w*l.w, striped_input, 0, NULL, NULL);

		int input_size = l.h*l.w;
		int output_size = l.out_h*l.out_w;
		clSetKernelArg(kernel_pool, 0, sizeof(cl_mem), &inbuf);
		clSetKernelArg(kernel_pool, 1, sizeof(cl_mem), &inbuf);
		clSetKernelArg(kernel_pool, 2, sizeof(int), &l.batch);
		clSetKernelArg(kernel_pool, 3, sizeof(int), &l.c);
		clSetKernelArg(kernel_pool, 4, sizeof(int), &input_size);
		clSetKernelArg(kernel_pool, 5, sizeof(int), &output_size);
		clSetKernelArg(kernel_pool, 6, sizeof(int), &l.w);



		cl_event kernel_event;
		cl_int status;
		//printf("Max pool launch kernel\n");
		status = clEnqueueNDRangeKernel(queue_coeff, kernel_pool, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
		checkError(status, "Failed to launch kernel");


		status = clFinish(queue_coeff);
		if (!hardware_enabled || l.fpga_save == 1)
			clEnqueueReadBuffer(queue_coeff, inbuf, CL_TRUE, 0, sizeof(float) * (l.c < STRIPES?STRIPES:l.c)*l.out_w*l.out_w, striped_output, 0, NULL, NULL);

	}
	else
	{
		 hw_maxpooling_fpga_x2_striped(striped_input,striped_output,
											   l.batch,
											   l.c,
											   l.h*l.w,
											   l.out_h*l.out_w,
											   l.w);
	}
	if (!hardware_enabled || l.fpga_load)
		alignedFree(striped_input);

	if (!hardware_enabled || l.fpga_save == 1)
	{
		remove_stripes(l.c,(l.out_w*l.out_w),striped_output,l.output);
		//clear_channel();
		alignedFree(striped_output);
	}

}



void SetBit(int layer,unsigned int *bits)
{
	int i = layer/32;
	int j = layer%32;
	unsigned int bits_before = bits[i];
	unsigned int mask = 0x1;
	for (int k = 0; k < j; k++)
		mask = mask << 1;
	bits[i] = bits_before | (mask);
}


extern "C" void forward_yolo_layer_fpga(const maxpool_layer l, network net)
{

#ifdef HARDWARE
	bool hardware_enabled = false;
#else
	bool hardware_enabled = false;
#endif

	if (hardware_enabled)
	{
	}
	else
	{
		printf("FPGA yolo layer2!\n");
		// Calculate layer mask
		int mask_width = l.c/8;
		unsigned int *layer_mask = (unsigned int *)malloc(mask_width); // bit mask
		for (int i = 0; i < mask_width/4; i++)
			layer_mask[i] = 0x0;
		{
			int n;
			for(n = 0; n < l.n; ++n){
				unsigned int start = ((n*(4+l.classes+1))+0);
				for (int s = start; s < (start+2); s++)
					SetBit(s,layer_mask);
				start = ((n*(4+l.classes+1))+4);
				for (int s = start; s < (start+l.classes+1); s++)
					SetBit(s,layer_mask);
			}
		}
		int c = l.c;
		if (c < STRIPES) c = STRIPES;
		if (c%STRIPES)
			c +=  (STRIPES-(c%STRIPES));
		float *striped_input = (float*)malloc(c*l.w*l.w*sizeof(float));
		float *striped_output = (float*)malloc(c*l.out_w*l.out_w*sizeof(float));
		stripe_input_data(l.c,l.w*l.w,net.input,striped_input);
		cl_mem in_mem = net.layers[l.id-1].fpga_outbuf;
		printf("yolo in_mem = %x\n",in_mem);
		printf("yolo out_mem = %x\n",l.fpga_outbuf);
		printf("yolo.c = %d\n",c);
		//if (l.fpga_load==1 || (l.first == 1))
			clEnqueueWriteBuffer(queue_activation, l.fpga_coeffbuf, CL_TRUE, 0, sizeof(int) * mask_width, layer_mask, 0, NULL, NULL);
		//if (l.fpga_load==1)
			clEnqueueWriteBuffer(queue_activation, in_mem, CL_TRUE, 0, sizeof(float) * c*l.w*l.h, striped_input, 0, NULL, NULL);

		//Set arguements
		clSetKernelArg(kernel_yolo, 0, sizeof(cl_mem), &l.fpga_coeffbuf); // mask!
		clSetKernelArg(kernel_yolo, 1, sizeof(cl_mem), &in_mem);
		clSetKernelArg(kernel_yolo, 2, sizeof(cl_mem), &l.fpga_outbuf);
		clSetKernelArg(kernel_yolo, 3, sizeof(cl_mem), &l.fpga_outbuf);
		clSetKernelArg(kernel_yolo, 4, sizeof(int), &l.outputs);
		clSetKernelArg(kernel_yolo, 5, sizeof(short), &l.batch);
		clSetKernelArg(kernel_yolo, 6, sizeof(short), &l.n);
		clSetKernelArg(kernel_yolo, 7, sizeof(short), &l.classes);
		clSetKernelArg(kernel_yolo, 8, sizeof(short), &l.w);
		clSetKernelArg(kernel_yolo, 9, sizeof(short), &l.h);
		clSetKernelArg(kernel_yolo, 10, sizeof(short), &c);
		cl_int status;

		//status = clEnqueueNDRangeKernel(queue_activation,kernel_yolo, 1, NULL, gSize, wgSize, 0, NULL, NULL);
  		checkError(status, "Failed to launch kernel");
		status = clFinish(queue_activation);

		// Reading output
		//if (l.fpga_save==1)
		clEnqueueReadBuffer(queue_activation, l.fpga_outbuf, CL_TRUE, 0, sizeof(float) * c*l.w*l.h, striped_output, 0, NULL, NULL);
	
		remove_stripes(c,(l.w*l.h),striped_output,l.output);
		free(striped_input);
		free(striped_output);
		free(layer_mask);
	}
	/*if (!hardware_enabled || l.fpga_load)
		alignedFree(striped_input);

	if (!hardware_enabled || l.fpga_save == 1)
	{
		remove_stripes(l.c,(l.out_w*l.out_w),striped_output,l.output);
		//clear_channel();
		alignedFree(striped_output);
	}*/

}


extern "C" void forward_avgpool_layer_fpga(const maxpool_layer l, network net)
{

    // Replace with FPGA optimised version
	float *striped_input = (float*)alignedMalloc((l.c < STRIPES?STRIPES:l.c)*l.w*l.w*sizeof(float)*2);

	stripe_input_data(l.c,l.w*l.w,net.input,striped_input);

	hw_average_pool_fpga_striped(striped_input,l.output,
    								   l.batch,
									   l.c,
    								   l.h*l.w,
    								   1.0f/(l.w*l.w));

	//clear_channel();
	alignedFree(striped_input);



}



extern "C" void activate_array_fpga(float* x, const int n, const ACTIVATION a)
{
	printf("activation\n");
	// data
	/*cl_int status;
   	int i;
   	unsigned int work_group_size = 1;
   	if (!initiailised)
   	{
   		//host_setup();
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
	else*/
	{
		// perform the activation using host code
		int i;
    		for(i = 0; i < n; ++i)
		{
        		//x[i] = activate(x[i], a);
    		}
	}
}



extern "C" void HardwareRunConvolution(
		float *input, float *coeffs, float *output,
		int ln_rounded,
		convolutional_layer l)
{
	int input_features = l.c;
	if (input_features < STRIPES)
		input_features = STRIPES;
	// RunOnHardware
	int y_div = 1;
	unsigned int InputBatchSize = (input_features*l.w*l.w)/STRIPES;
	//printf("InputBatchSize = %d\n",InputBatchSize);
	while (InputBatchSize > (MAX_INPUT_IMAGE_BATCH))
	{
		InputBatchSize = InputBatchSize >> 1;
		y_div = y_div*2;
		//printf("InputBatchSize = %d\n",InputBatchSize);
	}

	int batches_of_49 =  ((l.out_w*l.out_w)/(MAX_BATCH_SIZE))/y_div;
	cl_int status;
	// Write input to kernel global memory
	//printf("first = %d\n",l.first);
	if (l.first)	
		clEnqueueWriteBuffer(queue_coeff, l.fpga_coeffbuf, CL_TRUE, 0, sizeof(float) * l.size*l.size*input_features*l.n, coeffs, 0, NULL, NULL);
	
	if (l.fpga_load == 1)
		clEnqueueWriteBuffer(queue_convolve, inbuf, CL_TRUE, 0, sizeof(float) * l.w*l.w*input_features, input, 0, NULL, NULL);
      
	int i,j;
	//Set arguements
	clSetKernelArg(kernel_coeff_setup, 0, sizeof(cl_mem), &l.fpga_coeffbuf);
	clSetKernelArg(kernel_coeff_setup, 1, sizeof(int), &l.batch);
	clSetKernelArg(kernel_coeff_setup, 2, sizeof(int), &l.groups);
	clSetKernelArg(kernel_coeff_setup, 3, sizeof(int), &coeffs);
	clSetKernelArg(kernel_coeff_setup, 4, sizeof(int), &l.w);
	clSetKernelArg(kernel_coeff_setup, 5, sizeof(int), &l.out_w);
	clSetKernelArg(kernel_coeff_setup, 6, sizeof(int), &l.size);
	clSetKernelArg(kernel_coeff_setup, 7, sizeof(int), &l.pad);
	clSetKernelArg(kernel_coeff_setup, 8, sizeof(int), &input_features);
	clSetKernelArg(kernel_coeff_setup, 9, sizeof(int), &ln_rounded);
	clSetKernelArg(kernel_coeff_setup, 10, sizeof(int), &l.stride);
	clSetKernelArg(kernel_coeff_setup, 11, sizeof(int), &batches_of_49);
	clSetKernelArg(kernel_coeff_setup, 12, sizeof(int), &y_div);
	// Convolution
	clSetKernelArg(kernel_convolve, 0, sizeof(cl_mem), &inbuf);
	clSetKernelArg(kernel_convolve, 1, sizeof(cl_mem), &l.fpga_coeffbuf);
	clSetKernelArg(kernel_convolve, 2, sizeof(cl_mem), &outbuf);
	clSetKernelArg(kernel_convolve, 3, sizeof(int), &l.batch);
	clSetKernelArg(kernel_convolve, 4, sizeof(int), &l.groups);
	clSetKernelArg(kernel_convolve, 5, sizeof(int), &l.nweights);
	clSetKernelArg(kernel_convolve, 6, sizeof(int), &l.w);
	clSetKernelArg(kernel_convolve, 7, sizeof(int), &l.out_w);
	clSetKernelArg(kernel_convolve, 8, sizeof(int), &l.size);
	clSetKernelArg(kernel_convolve, 9, sizeof(int), &l.pad);
	clSetKernelArg(kernel_convolve, 10, sizeof(int), &l.c);
	clSetKernelArg(kernel_convolve, 11, sizeof(int), &ln_rounded);
	clSetKernelArg(kernel_convolve, 12, sizeof(int), &l.stride);
	clSetKernelArg(kernel_convolve, 13, sizeof(int), &batches_of_49);
	clSetKernelArg(kernel_convolve, 14, sizeof(int), &y_div);

	cl_event kernel_event;
	status = clEnqueueNDRangeKernel(queue_convolve, kernel_convolve, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");
	status = clEnqueueNDRangeKernel(queue_coeff, kernel_coeff_setup, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel");


	status = clFinish(queue_convolve);
	unsigned long start = 0;
	unsigned long end = 0;
	clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start,NULL);
	clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end,NULL);
	unsigned long duration = end - start;
	total_kernel_time += duration;
	//printf("kernel time = %f secs\n",duration*1e-9);
	//printf("total time = %f secs\n",total_kernel_time*1e-9);
	//clEnqueueReadBuffer(queue_convolve, outbuf, CL_TRUE, 0, sizeof(float) * l.out_w*l.out_w*l.n, output, 0, NULL, NULL);
	// Check queue has completed, in case we are not using a blocking function.
	//checkError(status, "Failed to finish");

	float div_sqrt_variance[2048];

	if (l.batch_normalize)
	for (int i = 0; i < l.out_c; i++)
	{
		div_sqrt_variance[i] = 1.0f/(sqrt(l.rolling_variance[i]) + .000001f);
	}
	int bnorm = l.batch_normalize?1:0;
	// activation and bias calculations
	int out_size = l.out_h*l.out_w;
	clSetKernelArg(kernel_activation, 0, sizeof(cl_mem), &outbuf);
	clSetKernelArg(kernel_activation, 1, sizeof(cl_mem), &inbuf);
	clSetKernelArg(kernel_activation, 2, sizeof(int), &l.batch);
	clSetKernelArg(kernel_activation, 3, sizeof(int), &l.out_c);
	clSetKernelArg(kernel_activation, 4, sizeof(int), &out_size);
	clSetKernelArg(kernel_activation, 5, sizeof(cl_mem), &l.fpga_div_sqrt_variance_buf);
	clSetKernelArg(kernel_activation, 6, sizeof(cl_mem), &l.fpga_rolling_mean_buf);
	clSetKernelArg(kernel_activation, 7, sizeof(cl_mem), &l.fpga_scales_buf);
	clSetKernelArg(kernel_activation, 8, sizeof(cl_mem), &l.fpga_biases_buf);
	clSetKernelArg(kernel_activation, 9, sizeof(int), &bnorm);
	//if ((l.n== 1024 ) && (l.c == 512))
	//	printf("break\n");
	if (l.first)
	{
	if (bnorm)
	{
		clEnqueueWriteBuffer(queue_activation, l.fpga_div_sqrt_variance_buf, CL_TRUE, 0, sizeof(float) * ln_rounded, div_sqrt_variance, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue_activation, l.fpga_rolling_mean_buf, CL_TRUE, 0, sizeof(float) * ln_rounded, l.rolling_mean, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue_activation, l.fpga_scales_buf, CL_TRUE, 0, sizeof(float) * ln_rounded, l.scales, 0, NULL, NULL);
	}
	clEnqueueWriteBuffer(queue_activation, l.fpga_biases_buf, CL_TRUE, 0, sizeof(float) * ln_rounded, l.biases, 0, NULL, NULL);
	}
	status = clEnqueueNDRangeKernel(queue_activation, kernel_activation, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
	//float *temp_results = new float[l.out_w*l.out_w*ln_rounded];
	status = clFinish(queue_activation);
	if (l.fpga_save)
	{
		clEnqueueReadBuffer(queue_activation, inbuf, CL_TRUE, 0, sizeof(float) * l.out_w*l.out_w*ln_rounded, output, 0, NULL, NULL);
	printf("total time = %f secs\n",total_kernel_time*1e-9);
	}
	checkError(status, "Failed to launch kernel");
	l.first = 0;

/*
	leaky_activate_fpga (output, output, l.batch,ln_rounded,l.out_h*l.out_w,
							  div_sqrt_variance,
							  l.rolling_mean,
							  l.scales,
							  l.biases,
							  l.batch_normalize?1:0);
	for (int i = 0; i <l.out_w*l.out_w*ln_rounded ;i++)
		if (temp_results[i] != output[i])
		{
			printf("MisMatch\n");
		}

	delete temp_results;*/

	return;
}

extern "C" void HardwareRunConvolutionBinary(	float *input, unsigned int *coeffs, float *output,
												float *scales,
												int ln_rounded,
												convolutional_layer l,
												int sub_block_size_x,
												int sub_block_size_y,
												int divx,
												int divy,
												network net)
{
	cl_int status;
	int no_sub_blocks = divx*divy;
	int input_features = l.c;
	if (input_features < STRIPES)
		input_features = STRIPES;

	// Write input to kernel global memory

	cl_mem in_mem;
	if (l.fpga_load)
	{
		in_mem = inbuf;
		clEnqueueWriteBuffer(queue_convolve, in_mem, CL_TRUE, 0, sizeof(float) * l.w*l.w*input_features, input, 0, NULL, NULL);
	}
	else
	{
		
		in_mem = net.layers[l.id-1].fpga_outbuf;
		/*printf("in_mem = %x\n",in_mem);
		float *temp = (float*)malloc(sizeof(float) * l.w*l.w*input_features);
		printf("size = l.w*l.w*input_features =%d\n",l.w*l.w*input_features); 
		clEnqueueReadBuffer(queue_convolve, in_mem, CL_TRUE, 0, sizeof(float) * l.w*l.w*input_features, temp , 0, NULL, NULL);
		int errors = 0;
		for (int i = 0; i <  l.w*l.w*input_features;i++) 
			if (errors < 34)
			if (temp[i] != input[i])
			{
				printf("%d error = %f != %f\n",i,temp[i],input[i]);	
				errors++;
			}
		//if (errors > 10)
		//	getchar();
		clEnqueueWriteBuffer(queue_convolve, in_mem, CL_TRUE, 0, sizeof(float) * l.w*l.w*input_features, input, 0, NULL, NULL);
		free(temp);*/
	}
	if (l.first == 1)
	{
		clEnqueueWriteBuffer(queue_convolve, l.fpga_scales_buf, CL_TRUE, 0, sizeof(float) * ln_rounded, scales, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue_coeff, l.fpga_coeffbuf, CL_TRUE, 0, sizeof(int) * l.size*l.size*ln_rounded*input_features/STRIPES, coeffs, 0, NULL, NULL);
	}

	int i,j;
	//Set arguements
	int not_used = 0;
	int one  = 1;
	clSetKernelArg(kernel_coeff_setup, 0, sizeof(cl_mem), &l.fpga_coeffbuf);
	clSetKernelArg(kernel_coeff_setup, 1, sizeof(int), &l.batch);
	clSetKernelArg(kernel_coeff_setup, 2, sizeof(int), &l.groups);
	clSetKernelArg(kernel_coeff_setup, 3, sizeof(int), &l.nweights);
	clSetKernelArg(kernel_coeff_setup, 4, sizeof(int), &l.w);
	clSetKernelArg(kernel_coeff_setup, 5, sizeof(int), &l.out_w);
	clSetKernelArg(kernel_coeff_setup, 6, sizeof(int), &l.size);
	clSetKernelArg(kernel_coeff_setup, 7, sizeof(int), &l.pad);
	clSetKernelArg(kernel_coeff_setup, 8, sizeof(int), &input_features);
	clSetKernelArg(kernel_coeff_setup, 9, sizeof(int), &ln_rounded);
	clSetKernelArg(kernel_coeff_setup, 10, sizeof(int), &l.stride);
	clSetKernelArg(kernel_coeff_setup, 11, sizeof(int), &one);
	clSetKernelArg(kernel_coeff_setup, 12, sizeof(int), &no_sub_blocks);
		

	int feature_size_in = l.w*l.w;
	int feature_size_out = l.out_w*l.out_w;
	int zero;
	clSetKernelArg(kernel_convolve, 0, sizeof(cl_mem), &in_mem);
	clSetKernelArg(kernel_convolve, 1, sizeof(cl_mem), &in_mem);
	clSetKernelArg(kernel_convolve, 2, sizeof(cl_mem), &l.fpga_outbuf);
	clSetKernelArg(kernel_convolve, 3, sizeof(unsigned short), &divy);
	clSetKernelArg(kernel_convolve, 4, sizeof(unsigned short), &divx);
	clSetKernelArg(kernel_convolve, 5, sizeof(int), &sub_block_size_x);
	clSetKernelArg(kernel_convolve, 6, sizeof(int), &sub_block_size_y);
	clSetKernelArg(kernel_convolve, 7, sizeof(int), &l.batch);
	clSetKernelArg(kernel_convolve, 8, sizeof(int), &l.groups);
	clSetKernelArg(kernel_convolve, 9, sizeof(int), &l.nweights);
	clSetKernelArg(kernel_convolve, 10, sizeof(int), &l.w);
	clSetKernelArg(kernel_convolve, 11, sizeof(int), &l.out_w);
	clSetKernelArg(kernel_convolve, 12, sizeof(int), &l.size);
	clSetKernelArg(kernel_convolve, 13, sizeof(int), &l.pad);
	clSetKernelArg(kernel_convolve, 14, sizeof(int), &l.c);
	clSetKernelArg(kernel_convolve, 15, sizeof(int), &ln_rounded);
	clSetKernelArg(kernel_convolve, 16, sizeof(int), &l.stride);
	clSetKernelArg(kernel_convolve, 17, sizeof(int), &zero);
	clSetKernelArg(kernel_convolve, 18, sizeof(int), &zero);
	clSetKernelArg(kernel_convolve, 19, sizeof(cl_mem), &l.fpga_scales_buf);
	clSetKernelArg(kernel_convolve, 20, sizeof(int), &zero);
	clSetKernelArg(kernel_convolve, 21, sizeof(int), &zero);
	clSetKernelArg(kernel_convolve, 22, sizeof(int), &zero);
	clSetKernelArg(kernel_convolve, 23, sizeof(int), &l.c);
	if (l.fpga_load || (l.first == 1))
	{
		float div_sqrt_variance[2048];

		if (l.batch_normalize)
		for (int i = 0; i < l.out_c; i++)
		{
			div_sqrt_variance[i] = 1.0f/(sqrt(l.rolling_variance[i]) + .000001f);
		}
		float activation_data[1024*4];
		for (int i = 0; i < l.out_c; i++) activation_data[i] = div_sqrt_variance[i];
		if ( l.batch_normalize)
		{
			for (int i = 0; i < l.out_c; i++) activation_data[i+l.out_c] = l.rolling_mean[i];
			for (int i = 0; i < l.out_c; i++) activation_data[i+l.out_c*2] = l.scales[i];
		}
		for (int i = 0; i < l.out_c; i++) activation_data[i+l.out_c*3] = l.biases[i];


		if (l.batch_normalize)
		for (int i = 0; i < l.out_c; i++)
		{
			div_sqrt_variance[i] = 1.0f/(sqrt(l.rolling_variance[i]) + .000001f);
		}
		clEnqueueWriteBuffer(queue_activation, l.fpga_biases_buf, CL_TRUE, 0, sizeof(float) * l.out_c * 4, activation_data, 0, NULL, NULL);
        }
	int bnorm = l.batch_normalize?1:0;
	// activation and bias calculations
	int out_size = l.out_h*l.out_w;
	int activation = ((l.activation == LEAKY)||(l.activation == BINARY))? FPGA_LEAKY:FPGA_LINEAR;
	clSetKernelArg(kernel_convolve, 24, sizeof(cl_mem), &l.fpga_biases_buf);
	clSetKernelArg(kernel_convolve, 25, sizeof(int), & l.batch_normalize );
	clSetKernelArg(kernel_convolve, 26, sizeof(int), &activation );

	cl_event kernel_event;
	status = clEnqueueNDRangeKernel(queue_coeff, kernel_coeff_setup, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");

	status = clEnqueueNDRangeKernel(queue_convolve, kernel_convolve, 1, NULL, gSize, wgSize, 0, NULL,  &kernel_event);
	checkError(status, "Failed to launch kernel");

	status = clWaitForEvents(1,&kernel_event);
	
	unsigned long start = 0;
	unsigned long end = 0;
	clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start,NULL);
	clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end,NULL);
	unsigned long duration = end - start;
	total_kernel_time += duration;
	printf("kernel time = %f secs\n",duration*1e-9);
	printf("total time = %f secs\n",total_kernel_time*1e-9);


	//status = clEnqueueNDRangeKernel(queue_activation, kernel_activation, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
	//float *temp_results = new float[l.out_w*l.out_w*ln_rounded];
	//status = clFinish(queue_activation);
	//clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start,NULL);
	//clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end,NULL);
	//duration = end - start;
	//printf("kernel time = %f secs\n",duration*1e-9);
	//total_kernel_time += duration;

	if (l.fpga_save)
		clEnqueueReadBuffer(queue_activation, l.fpga_outbuf, CL_TRUE, 0, sizeof(float) * feature_size_out*ln_rounded, output, 0, NULL, NULL);
	// Check queue has completed, in case we are not using a blocking function.
	checkError(status, "Failed to finish");
	return;
}




void remove_stripes(int in_f,int size,float *in,float *out,unsigned int offset)
{
	unsigned int c=0;
	int j,i,k;
	for ( j = 0; j < in_f; j+=STRIPES)
	for ( i = 0; i < size; i++)
	{
		for ( k = 0; k < STRIPES; k++)
		{
			if ((j+k) < in_f)
				out[offset + (size*(j+k))+i] = in[c];
			c++;
		}
	}
}

void route_stripe_input_data(int in_f,int size,float *in,float *out)
{
	int i,j,k;
	unsigned int c= 0;
	for ( j = 0; j < in_f; j+=STRIPES)
	for ( i = 0; i < size; i++)
	{
		for ( k = 0; k < STRIPES; k++)
		{
			float val;
			if ((j+k) >= in_f)
				val = 0;
			else
				val = in[(size*(j+k))+i];
			out[(size*j)+(i<<STRIPES_DIV)+k] = val;
		}
	}
}

void route_layer_fpga_striped(float *inbuf1,
		  float *inbuf2,
		  float *outbuf,
		  unsigned int features1,
		  unsigned int feature_size1,
		  unsigned int features2,
		  unsigned int feature_size2,
		  // Output stripe feature size
		  unsigned int features3,
		  unsigned int feature_size3)
{
	
	remove_stripes(features1,feature_size1,inbuf1,outbuf,0);
	if (features2)
	remove_stripes(features2,feature_size2,inbuf2,outbuf,(features1*feature_size1));
	route_stripe_input_data(features3,feature_size3,outbuf,outbuf);
}



extern "C" void forward_route_layer_fpga(const route_layer l, network net)
{
	printf("FPGA route layer\n");
	printf("mem out = %x\n",l.fpga_outbuf);
#ifdef HARDWARE
	bool hardware_enabled = true;
#else
	bool hardware_enabled = false;
#endif
	if (hardware_enabled)
	{
	route_layer a,b,c;
	a = net.layers[l.input_layers[0]];
	b = net.layers[l.input_layers[1]];
	c = net.layers[l.id+1]; // Next layer
	unsigned int features1 = a.c;
	unsigned int feature_size1 = a.out_w*a.out_w;
	unsigned int features2 = l.n == 2 ? b.c:0;
	unsigned int feature_size2 = l.n == 2? b.out_w*b.out_w:0;	
	unsigned int features3 =  l.n == 2 ?c.c:a.c;
	unsigned int feature_size3 = l.n == 2 ? c.out_w*c.out_w: a.out_w*a.out_w;

   	unsigned int layer_size1 = l.input_sizes[0];
    	unsigned int layer_size2 = l.n==2?l.input_sizes[1]:0;
    	float *input0_data = net.layers[l.input_layers[0]].output;
   	float *input1_data = l.n==2?net.layers[l.input_layers[1]].output:0;
	cl_mem mem0 = net.layers[l.input_layers[0]].fpga_outbuf;
	cl_int status;
	cl_mem mem1 = l.n==2?net.layers[l.input_layers[1]].fpga_outbuf:NULL;

	if (!hardware_enabled || (l.fpga_load == 1))
	{
		float *unstriped_input1 = (float*)malloc(features1*feature_size1*sizeof(float));
		float *unstriped_input2 = (float*)malloc(features2*feature_size2*sizeof(float));

		route_stripe_input_data(features1,feature_size1,a.output,unstriped_input1);
		clEnqueueWriteBuffer(queue_activation, mem0 , CL_TRUE, 0, sizeof(float) * (layer_size1), unstriped_input1 , 0, NULL, NULL);
		if (mem1)
		{
			route_stripe_input_data(features2,feature_size2,b.output,unstriped_input2);
			clEnqueueWriteBuffer(queue_activation, mem1 , CL_TRUE, 0, sizeof(float) * (layer_size2), unstriped_input2 , 0, NULL, NULL);
		}
		free(unstriped_input1);
		free(unstriped_input2);
	}
	clSetKernelArg(kernel_route, 0, sizeof(cl_mem), &mem0 );
	clSetKernelArg(kernel_route, 1, sizeof(cl_mem), l.n==2?&mem1:&mem0 );
	clSetKernelArg(kernel_route, 2, sizeof(cl_mem), &l.fpga_outbuf);
	clSetKernelArg(kernel_route, 3, sizeof(int), &layer_size1);
	clSetKernelArg(kernel_route, 4, sizeof(int), &layer_size2);
	clSetKernelArg(kernel_route, 5, sizeof(int), &l.n);
	cl_event kernel_event;
	status = clEnqueueNDRangeKernel(queue_activation, kernel_route, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);

	status = clFinish(queue_activation);
	




	if (!hardware_enabled || (l.fpga_save == 1))
	{		
		float *striped_output = (float*)malloc((layer_size1+layer_size2)*sizeof(float));
		clEnqueueReadBuffer(queue_activation, l.fpga_outbuf, CL_TRUE, 0, sizeof(float) * (layer_size1+layer_size2),striped_output , 0, NULL, NULL);
		a = net.layers[l.input_layers[0]];
		printf("features = %d\n",features3);
		printf("feature_size = %d\n",feature_size3);
		remove_stripes(features3,feature_size3,striped_output,l.output,0);

		free(striped_output);
	}
	// Check queue has completed, in case we are not using a blocking function.
	checkError(status, "Failed to finish");
	}
	else
	{
	   	unsigned int layer_size1 = l.input_sizes[0];
	    	unsigned int layer_size2 = l.n==2?l.input_sizes[1]:0;
	    	float *input0_data = net.layers[l.input_layers[0]].output;
	   	float *input1_data = l.n==2?net.layers[l.input_layers[1]].output:0;

		route_layer_fpga(input0_data,
				input1_data,
					 l.output,
					 layer_size1,
					 layer_size2,
					 l.n
					 );
	}
	return;

#ifdef OLD

	float *unstriped_output = (float*)malloc(features3*feature_size3*sizeof(float));

	route_stripe_input_data(features1,feature_size1,a.output,unstriped_input1);
	getchar();
	if (l.n == 2)
		route_stripe_input_data(features2,feature_size2,b.output,unstriped_input2);
	
	printf("b.c = %d\n",b.c);
	printf("b.n = %d\n",b.n);
	printf("features2 = %d\n",features2);
	printf("features2 = %d\n",features2);
	printf("features_size2 = %d\n",feature_size2);
	printf("features1 = %d\n",features1);
	printf("features3 = %d\n",features3);
	printf("features_size1 = %d\n",feature_size1);
	printf("features_size3 = %d\n",feature_size3);
	printf("unstriped_input2 = %x\n",unstriped_input2);
	printf("b.output = %x\n",b.output);
	
	route_layer_fpga_striped(
		  unstriped_input1,
		  unstriped_input2,
		  unstriped_output,
		  features1,
		  feature_size1,
		  features2,
		  feature_size2,
		  // Output stripe feature size
		  features3,
		  feature_size3);

	remove_stripes(features3,feature_size3,unstriped_output,l.output,0);


	free(unstriped_input1);
	if (feature_size2)
		free(unstriped_input2);
	free(unstriped_output);

	return;


    // Route layer uses other parts of the network for data and has no input of it's own, only output

	if (hardware_enabled)
	{
		// Find input layers
		cl_mem input0,input1;
		cl_int status;
		cl_event kernel_event;
	
		input0 = net.layers[l.input_layers[0]].fpga_outbuf;
		input1 = net.layers[l.input_layers[1]].fpga_outbuf;
		printf("input0 w = %d\n",net.layers[l.input_layers[0]].w);
		printf("input0 out w = %d\n",net.layers[l.input_layers[0]].out_w);
		printf("input1 out w = %d\n",net.layers[l.input_layers[1]].w);
		printf("input1 w = %d\n",net.layers[l.input_layers[1]].out_w);
		printf("input0 l.c = %d\n",net.layers[l.input_layers[0]].c);
		printf("input1 l.c = %d\n",net.layers[l.input_layers[1]].c);
	        printf("net.layers[l.input_layers[0]].type = %d\n",net.layers[l.input_layers[0]].type);
	        printf("net.layers[l.input_layers[1]].type = %d\n",net.layers[l.input_layers[1]].type);
	    	unsigned int layer_size1 = l.input_sizes[0];
	    	unsigned int layer_size2 = l.n==2?l.input_sizes[1]:0;
		printf("input0 = %x\n",input0);
		printf("input1 = %x\n",input1);
		printf("layer_size1= %x\n",layer_size1);
		printf("layer_size2= %x\n",layer_size2);

		printf("l.fpga_outbuf= %x\n",l.fpga_outbuf);
		int l_c0 = net.layers[l.input_layers[0]].c;
		int l_c1 = net.layers[l.input_layers[1]].c;
		int l_w0 = net.layers[l.input_layers[0]].out_w;
		int l_w1 = net.layers[l.input_layers[1]].out_w;


		l_c0 = l_c0 < STRIPES? STRIPES: l_c0;
		l_c1 = l_c1 < STRIPES? STRIPES: l_c1;

		//float *unstriped_input1 = (float*)alignedMalloc(sizeof(float) * (l_c0 * l_w0 * l_w0));
		//float *unstriped_input2 = (float*)alignedMalloc(sizeof(float) * (l_c1 * l_w1 * l_w1));
		
		//if (!hardware_enabled || (l.fpga_save == 1))
	    	float *input0_data = net.layers[l.input_layers[0]].output;
    		float *input1_data = l.n==2?net.layers[l.input_layers[1]].output:0;
		// Read data from previeous frames
		//clEnqueueReadBuffer(queue_activation, input0, CL_TRUE, 0, sizeof(float) * ((l_c0 * l_w0 * l_w0)), input0_data , 0, NULL, NULL);
		//if (layer_size2 != 0)
		//	clEnqueueReadBuffer(queue_activation, input1, CL_TRUE, 0, sizeof(float) * ( (l_c1 * l_w1 * l_w1)), input1_data , 0, NULL, NULL);
		//
		//remove_stripes(l_c0,l_w0*l_w0,input0_data,unstriped_input1);
		//if (layer_size2 != 0)
		//	remove_stripes(l_c1,l_w1*l_w1,input1_data,unstriped_input2);
		
		printf("l.n = %d\n",l.n);
		clEnqueueWriteBuffer(queue_activation, input0, CL_TRUE, 0, sizeof(float) * (layer_size1), input0_data , 0, NULL, NULL);
		clEnqueueWriteBuffer(queue_activation, input1, CL_TRUE, 0, sizeof(float) * (layer_size2), input1_data , 0, NULL, NULL);
	

		/*int errors = 0;
		for (int i = 0; i < layer_size1; i ++)
			if (unstriped_input1[i] != net.layers[l.input_layers[0]].output[i])
			{
				printf("error %d : %f not %f : input0_data=%f\n",i,unstriped_input1[i],net.layers[l.input_layers[0]].output[i],input0_data[i]);
				errors++;
				if (errors > 64) break;
			}		*/

		    	/*route_layer_fpga(unstriped_input1,
	    					unstriped_input2,
							 l.output,
							 layer_size1,
							 layer_size2,
							 l.n
							 );*/


		clSetKernelArg(kernel_route, 0, sizeof(cl_mem), &input0);
		clSetKernelArg(kernel_route, 1, sizeof(cl_mem), &input1);
		clSetKernelArg(kernel_route, 2, sizeof(cl_mem), &l.fpga_outbuf);
		clSetKernelArg(kernel_route, 3, sizeof(int), &layer_size1);
		clSetKernelArg(kernel_route, 4, sizeof(int), &layer_size2);
		clSetKernelArg(kernel_route, 5, sizeof(int), &l.n);
	
		status = clEnqueueNDRangeKernel(queue_activation, kernel_route, 1, NULL, gSize, wgSize, 0, NULL, &kernel_event);
	
		status = clFinish(queue_activation);
		float *striped_output;
		//if (!hardware_enabled || (l.fpga_save == 1))
		{
		//	striped_output = (float*)alignedMalloc((layer_size1+layer_size2)*sizeof(float));
			clEnqueueReadBuffer(queue_activation, l.fpga_outbuf, CL_TRUE, 0, sizeof(float) * (layer_size1+layer_size2), l.output , 0, NULL, NULL);
		}
		// Check queue has completed, in case we are not using a blocking function.
		checkError(status, "Failed to finish");
		printf("free arrays\n");
		//printf("unstriped_input1 = %x\n",unstriped_input1);
		//printf("unstriped_input2 = %x\n",unstriped_input2);
		//alignedFree(unstriped_input1);

		//if (layer_size2)
		//	alignedFree(unstriped_input2);
	
	}
	else
	{
	    // Assuming route size of two
	    if ((l.n != 2) && (l.n != 1))
	    {
	    	assert("Unsupported number of routes");
	    }
	    else
	    {

	    	unsigned int layer_size1 = l.input_sizes[0];
	    	unsigned int layer_size2 = l.n==2?l.input_sizes[1]:0;
	    	float *input0 = net.layers[l.input_layers[0]].output;
	    	float *input1 = l.n==2?net.layers[l.input_layers[1]].output:0;

	    	route_layer_fpga(input0,
	    					 input1,
							 l.output,
							 layer_size1,
							 layer_size2,
							 l.n
							 );
	    }
	}
#endif
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
		clReleaseKernel(kernel_activation);
	}
	if (kernel_pool) {
		clReleaseKernel(kernel_pool);
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

