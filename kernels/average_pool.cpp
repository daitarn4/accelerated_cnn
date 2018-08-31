/*
 * max_pooling.cpp
 *
 *  Created on: Feb 19, 2018
 *      Author: rchamberlain
 *  To save on space using a fixed sized pooling layer as darknet always a factor of 2 reduction.
 *  This is nearly always the case from most networks.
 *
 *  Also output of convolution is striped and this must also be handled.
 */

#include "basic_convolution_striped.h"

#ifdef ALTERA_CL
__kernel
void hw_average_pool_fpga_striped(__global float *restrict inbuf, __global float *restrict outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   float div_filter_size)
#else
void hw_average_pool_fpga_striped(float* inbuf, float* outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   float div_filter_size)
#endif
{
    // Use sliding window for max pooling layer
	printf("FPGA average pool layer\n");
    for(int b = 0; b < batch; ++b){
    	for (int i_f = 0; i_f < in_f; i_f += STRIPES)
    	{
    		// Handle stripes accordingly
    		int d =  (b*in_f) + i_f;
    		for (int s = 0; s < STRIPES; s++)
    		{
    			float sum = 0;
    			int c = s + ((i_f)*filter_size);
    			for (int i = 0; i < filter_size; i++)
    			{
    				int address = c;
    				float val = inbuf[address];
    				sum += val;
    				c+=STRIPES;
    			}
    			sum *= div_filter_size;
    			outbuf[d+s] = sum; // outputs are now aligned and stripes can be ignored.
    		}
    	}
    }
}
