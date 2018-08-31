/*
 * basic_convolution_striped.h
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_BASIC_CONVOLUTION_STRIPED_H_
#define KERNELS_BASIC_CONVOLUTION_STRIPED_H_



#ifndef ALTERA_CL
#include <stdio.h>
#include <malloc.h>
#endif

#define PAR_PIX 2
#define PAR_PIX_SHIFT 1
#define PAR_PIX_MASK 0x1

#define STRIPES_32
#ifdef STRIPES_16
	#define STRIPES 16
	#define STRIPES_DIV 4
	#define STRIPES_BIT_MASK 0xffff
	#define STRIPES_BITS 16
#endif
#ifdef STRIPES_32
	#define STRIPES 32
	#define STRIPES_DIV 5
	#define STRIPES_BIT_MASK 0xffffffff
	#define STRIPES_BITS 32
#endif
#define BINARY_FLOAT_SCALE 10

// Darknet 19 batchsize
//#define MAX_BATCH_SIZE (49)
// YOLO v2/v3 batchsize
#define MAX_BATCH_SIZE (512)
#define MAX_PADDING_SPACE (416*6)
// Max size set to minimum whole subdivision of biggest conv layer
// Adding padding
#define MAX_INPUT_IMAGE_BATCH (((52+2)*(52+2)) * (32/STRIPES))
#define MAX_OUTPUT_IMAGE_BATCH (((52)*(52)) * (32/STRIPES))
//#define MAX_INPUT_IMAGE_BATCH (((52+2)*(52+2)) * (1))
//#define MAX_OUTPUT_IMAGE_BATCH (((52)*(52)) * (1))

enum {FPGA_LEAKY=1,FPGA_LINEAR=2};

#ifdef ALTERA_CL
__kernel
#endif
void basic_convolution_striped_load_coeffs_kernel_standard(
#ifdef ALTERA_CL
					   __global float *restrict coeffs,
#else
					   float *coeffs,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int out_f,
					   int stride);

#ifdef ALTERA_CL
__kernel
#endif
void basic_convolution_striped_load_coeffs_kernel(
#ifdef ALTERA_CL
					   __global float *restrict coeffs,
#else
					   float *coeffs,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int out_f,
					   int stride);


#ifdef ALTERA_CL
__kernel
#endif
void basic_convolution_striped_kernel(
#ifdef ALTERA_CL
					   __global float *restrict input,
					   __global float *restrict coeffs,
					   __global float *restrict output,
#else
					   float *input, float *coeffs, float *output,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int out_f,
					   int stride);


#ifdef ALTERA_CL
__kernel
void hw_maxpooling_fpga_x2_striped(__global float *restrict inbuf, __global float *restrict outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   int output_filter_size,
								   int size_x);
#else
void hw_maxpooling_fpga_x2_striped(float* inbuf, float* outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   int output_filter_size,
								   int size_x);
#endif

#ifdef ALTERA_CL
__kernel
void hw_average_pool_fpga_striped(__global float *restrict inbuf, __global float *restrict outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   float div_filter_size);
#else
void hw_average_pool_fpga_striped(float* inbuf, float* outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   float div_filter_size);
#endif
#ifdef __cplusplus
    extern "C" {
#endif

#ifdef ALTERA_CL
__kernel
void leaky_activate_fpga (__global float *restrict inbuf, __global float *restrict outbuf, int batch,int out_c,int size,
		__global float *restrict data,
		/*__global float *restrict div_sqrt_variance,
		__global float *restrict rolling_mean,
		__global float *restrict scales,
		__global float *restrict biases,*/int batch_normalised,
		 int activation);
#else
void leaky_activate_fpga (float *inbuf, float *outbuf, int batch,int out_c,int size,
						  float *data,
						  //float *div_sqrt_variance,
						  //float *rolling_mean,
						  //float *scales,
						  //float *biases,
						  int batch_normalised,
						  int activation);
#endif





/*
 * Helper functions
 */
    typedef struct
    {
     	float coeffs[STRIPES];
    }coeff_vector;

#ifndef ALTERA_CL
void stripe_coefficients(int in_f,int out_f, int kernel_size,float *in,float *out);
void stripe_input_data(int in_f,int size,float *in,float *out);
void remove_stripes(int in_f,int size,float *in,float *out);
void clear_channel();
#endif

#ifdef __cplusplus
    }
#endif
#ifndef ALTERA_CL
void write_channel_intel(coeff_vector *chan,coeff_vector value);
coeff_vector read_channel_intel(coeff_vector *chan);

// Declare producer consimer channels
//extern CHAN(float,16) leaky_activate_input_channel;
//extern CHAN(float,16) leaky_activate_output_channel;

#endif


#endif /* KERNELS_BASIC_CONVOLUTION_STRIPED_H_ */
