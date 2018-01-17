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

#define STRIPES 32
#define STRIPES_DIV 5

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



/*
 * Helper functions
 */
#ifdef __cplusplus
    extern "C" {
#endif
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
#endif

#endif /* KERNELS_BASIC_CONVOLUTION_STRIPED_H_ */
