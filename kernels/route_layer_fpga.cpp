/*
 * route_layer_fpga.cpp
 *
 *  Created on: Apr 16, 2018
 *  Author: rchamberlain
 *  Appends to layers to create new layer
 */


#include "route_layer_fpga.h"

#ifdef ALTERA_CL
__kernel
void route_layer_fpga(__global float *restrict inbuf1,
					  __global float *restrict inbuf2,
					  __global float *restrict outbuf,
					  unsigned int feature_size1,
					  unsigned int feature_size2,
					  unsigned int n)
#else
void route_layer_fpga(float *inbuf1,
		  float *inbuf2,
		  float *outbuf,
		  unsigned int feature_size1,
		  unsigned int feature_size2,
		  unsigned int n)
#endif
{
	int count = 0;

	
	for (int i = 0; i < feature_size1; i+=STRIPES)
	{
		#pragma unroll
		for (int s = 0; s < STRIPES; s++)
			outbuf[count++] = inbuf1[i+s];
	}
	if (n == 2)
	for (int i = 0; i < feature_size2; i+=STRIPES)
	{
		#pragma unroll
		for (int s = 0; s < STRIPES; s++)
			outbuf[count++] = inbuf2[i+s];
	}
}
