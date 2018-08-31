/*
 * basic_convolution_striped.h
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_BASIC_CONVOLUTION_STRIPED_BINARY_H_
#define KERNELS_BASIC_CONVOLUTION_STRIPED_BINARY_H_


#include "basic_convolution_striped.h"

#ifndef ALTERA_CL
#include "math.h"
#define half_ float
#else
#endif

#ifdef __cplusplus
    extern "C" {
#endif


#ifdef ALTERA_CL
__kernel
#else
#endif
void basic_convolution_striped_load_coeffs_kernel_binary_bits(
#ifdef ALTERA_CL
					   __global unsigned int *restrict coeffs,
#else
					   unsigned int *coeffs,
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
					   int y_div);


#ifdef ALTERA_CL
__kernel
#else
#endif
void basic_convolution_striped_load_coeffs_kernel_binary(
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
					   int y_div);

#ifdef ALTERA_CL
__kernel
#else
#endif
void basic_convolution_striped_kernel_binary(
#ifdef ALTERA_CL
					   __global float *restrict input,
					   __global float *restrict scale, // binary output scaler
					   __global float *restrict output,
#else
					   float *input, float *scale, float *output,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int size_div_y_div,
					   int out_size_div_y_div,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int true_in_f, // true size if scaled
					   int out_f,
					   int stride,
					   int batches_of_49,
					   int y_div); // Reduces the amount of output cahce memory required

#ifdef __cplusplus
    }
#endif


#endif /* KERNELS_BASIC_CONVOLUTION_STRIPED_H_ */
