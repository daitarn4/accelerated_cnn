/*
 * upsample_x2_fpga.h
 *
 *  Created on: May 15, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_UPSAMPLE_X2_FPGA_H_
#define KERNELS_UPSAMPLE_X2_FPGA_H_

#include "basic_convolution_striped.h"

#ifdef __cplusplus
    extern "C" {
#endif
#ifdef ALTERA_CL
__kernel
void upsample_x2_fpga(__global float *restrict inbuf, __global float *restrict outbuf,
		   	   	   	   	   	   unsigned int batch,
							   unsigned int in_f,
							   unsigned int output_filter_size,
							   unsigned short out_w,
							   unsigned short out_h,
							   unsigned int stripe_input_block_size,
							   unsigned int stripe_output_block_size);
#else
void upsample_x2_fpga(float* inbuf, float* outbuf,
								   unsigned int batch,
								   unsigned int in_f,
								   unsigned int output_filter_size,
								   unsigned short out_w,
								   unsigned short out_h,
								   unsigned int stripe_input_block_size,
								   unsigned int stripe_output_block_size);
#endif
#ifdef __cplusplus
    }
#endif

#endif /* KERNELS_UPSAMPLE_X2_FPGA_H_ */
