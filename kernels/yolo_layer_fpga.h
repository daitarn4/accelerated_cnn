/*
 * yolo_layer_fpga.h
 *
 *  Created on: May 15, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_YOLO_LAYER_FPGA_H_
#define KERNELS_YOLO_LAYER_FPGA_H_

#include "basic_convolution_striped.h"
#ifdef __cplusplus
    extern "C" {
#endif

#ifdef ALTERA_CL
__kernel
void yolo_layer_fpga(__global unsigned int *restrict layer_mask,
					 __global float *restrict in,
					 __global float *restrict out,
					 __global float *restrict out2,
					 int input_block_size,
					 short batch, short l_n,short classes,short w, short h,short c
					 );
#else
#include "math.h"
void yolo_layer_fpga(unsigned int *layer_mask,float *in, float *out, float  *out2, int input_block_size,
					     short batch, short l_n,short classes,short w, short h,short c);
#endif
#ifdef __cplusplus
    }
#endif

#endif /* KERNELS_YOLO_LAYER_FPGA_H_ */
