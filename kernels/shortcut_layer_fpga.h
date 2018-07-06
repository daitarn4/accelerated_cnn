/*
 * shortcut_layer_fpga.h
 *
 *  Created on: May 14, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_SHORTCUT_LAYER_FPGA_H_
#define KERNELS_SHORTCUT_LAYER_FPGA_H_

#include "basic_convolution_striped.h"

#ifdef __cplusplus
    extern "C" {
#endif
#ifdef ALTERA_CL
__kernel
void shortcut_layer_fpga(int input_block_size,
					     int batch, int w1, int h1, int c1,
						 __global volatile float *restrict add,
						 int w2, int h2, int c2, float s1, float s2,
						 __global volatile float *restrict in,
						 __global volatile float *restrict out,
						 __global volatile float *restrict out2 // Same as out, but allows pipelining!
						 );
#else
void shortcut_layer_fpga(int input_block_size,
				  int batch, int w1, int h1, int c1,
				  float *add, int w2, int h2, int c2, float s1, float s2,
				  float *in,float *out,float *out2);
#endif

#ifdef __cplusplus
    }
#endif

#endif /* KERNELS_SHORTCUT_LAYER_FPGA_H_ */
