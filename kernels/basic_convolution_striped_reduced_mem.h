/*
 * basic_convolution_striped.h
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_BASIC_CONVOLUTION_STRIPED_REDUCED_MEM_H_
#define KERNELS_BASIC_CONVOLUTION_STRIPED_REDUCED_MEM_H_


#include "basic_convolution_striped.h"

// Stay within a single block ram
#define MAX_BATCH_SIZE (49)
#define MAX_PADDING_SPACE (512*3)
#define MAX_INPUT_IMAGE_BATCH ((8192)-MAX_PADDING_SPACE)

#ifdef __cplusplus
    extern "C" {
#endif

#ifdef ALTERA_CL
__kernel
#endif
void basic_convolution_striped_load_coeffs_kernel_reduced_mem(
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
					   int stride,
					   int batches_of_49,
					   int div_y);
					   //int input_batches);

#ifdef ALTERA_CL
__kernel
#endif

void basic_convolution_striped_kernel_reduced_mem(
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
					   int stride,
					   int batches_of_49,
					   int div_y);
					   //int input_batches); // Reduces the amount of output cache memory required

#ifdef __cplusplus
    }
#endif


#endif /* KERNELS_BASIC_CONVOLUTION_STRIPED_H_ */
