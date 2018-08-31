/*
 * basic_convolution_striped.h
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 */

#ifndef CONV_BINARY_FPGA_H_
#define CONV_BINARY_FPGA_H_


#include "basic_convolution_striped.h"

typedef struct {int s[STRIPES];} vec_int;
typedef struct {unsigned int s[STRIPES];} vec_uint;
typedef struct {short s[STRIPES];} vec_short;
typedef struct {float s[STRIPES];} vec_float;

#ifdef __cplusplus
    extern "C" {
#endif

#ifdef ALTERA_CL
__kernel
#else
#endif
void conv_coeffs_binary_subblock_fpga(
#ifdef ALTERA_CL
					   __global vec_uint *restrict coeffs,
#else
					   vec_uint *coeffs,
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
					   int stride,
					   int batches_of_49,
					   int y_div,
					   int block_size);


#ifdef ALTERA_CL
__kernel
#endif

#ifdef ALTERA_CL
__kernel
#else
//extern "C"
#endif
void conv_binary_subblock_fpga_v4(
#ifdef ALTERA_CL
					   __global vec_float  *restrict input,
					   __global unsigned int *restrict coeffs,
					   __global float *restrict output,
#else
					   vec_float *input, unsigned int *coeffs, float *output,
#endif
					   unsigned short no_sub_blocks_y,
					   unsigned short no_sub_blocks_x,
					   int sub_block_width,
					   int sub_block_height,

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
					   int y_div,
#ifdef ALTERA_CL
					   __global float *restrict binary_scale,
#else
					   float *binary_scale,
#endif
					   int batch_size,
					   int step_size,
					   int out_step_size,
					   int l_c,
					   unsigned int total_block_count,
					   unsigned char power
					  ) ; // Reduces the amount of output cahce memory required
					   //int input_batches); // Reduces the amount of output cache memory required


// Output kernel
#ifdef ALTERA_CL
__kernel
#else
//extern "C"
#endif
void conv_activations_v4(
#ifdef ALTERA_CL
					   __global vec_float *restrict output,
#else
					   float *output,
#endif
					   unsigned short no_sub_blocks_y,
					   unsigned short no_sub_blocks_x,
					   int sub_block_width,
					   int sub_block_height,

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
					   int y_div,
#ifdef ALTERA_CL
					   __global float *restrict binary_scale,
#else
					   float *binary_scale,
#endif
					   int batch_size,
					   int step_size,
					   int out_step_size,
					   int l_c,
					   // adding activation to convolution
#ifdef ALTERA_CL
					  __global float *restrict data,
#else
					  float *data,
#endif
					  int batch_normalised,
					  int activation

					  );

#ifdef __cplusplus
    }
#endif


#endif /* KERNELS_BASIC_CONVOLUTION_STRIPED_H_ */
