/*
 * route_layer_fpga.h
 *
 *  Created on: Apr 16, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_ROUTE_LAYER_FPGA_H_
#define KERNELS_ROUTE_LAYER_FPGA_H_

#include "basic_convolution_striped.h"

#ifdef ALTERA_CL
__kernel
void route_layer_fpga(__global float *restrict inbuf1,
					  __global float *restrict inbuf2,
					  __global float *restrict outbuf,
					  unsigned int feature_size1,
					  unsigned int feature_size2,
					  unsigned int n);
#else
void route_layer_fpga(float *inbuf1,
		  float *inbuf2,
		  float *outbuf,
		  unsigned int feature_size1,
		  unsigned int feature_size2,
		  unsigned int n);
#endif



#endif /* KERNELS_ROUTE_LAYER_FPGA_H_ */
