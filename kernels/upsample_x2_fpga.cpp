/*
 * upsample_x2_fpga.cpp
 *
 *  Created on: May 15, 2018
 *      Author: rchamberlain
 */

#include "upsample_x2_fpga.h"


#ifdef ALTERA_CL
__kernel
void upsample_x2_fpga(__global float *restrict inbuf, __global float *restrict outbuf,
		   	   	   	   	   	   unsigned int batch,
							   unsigned int in_f,
							   unsigned int output_filter_size,
							   unsigned short out_w,
							   unsigned short out_h,
							   unsigned int stripe_input_block_size,
							   unsigned int stripe_output_block_size)
#else
void upsample_x2_fpga(float* inbuf, float* outbuf,
								   unsigned int batch,
								   unsigned int in_f,
								   unsigned int output_filter_size,
								   unsigned short out_w,
								   unsigned short out_h,
								   unsigned int stripe_input_block_size,
								   unsigned int stripe_output_block_size)
#endif
{
	printf("upsample fpga\n");
	unsigned int stripes_input_offset = 0;
	unsigned int stripes_output_offset = 0;
    for(unsigned short b = 0; b < batch; ++b){
    	for (unsigned short i_f = 0; i_f < in_f; i_f += STRIPES)
    	{
			unsigned short x,y;
			unsigned short x_out,y_out;
			x_out = y_out = 0;

			for (unsigned int i = 0; i < (output_filter_size); i+= 1)
			{
				x = x_out >> 1;
				y = y_out >> 1;
				unsigned int address = x + (((out_w>>1)&0xffff)*(y&0xffff))&0xffffffff; // Forces single DSP use
				unsigned int out_address = x_out + ((out_w&0xffff)*(y_out&0xffff))&0xffffffff; // Forces single DSP use
				#pragma unroll
				for (int p = 0; p < STRIPES; p++)
				{
					float val = inbuf[stripes_input_offset + (address<<STRIPES_DIV) + p];
					outbuf[stripes_output_offset + (out_address<<STRIPES_DIV) + p] = val;
				}
				x_out = x_out != (out_w-1)? x_out+1:0;
				y_out = x_out == 0 ? y_out +1: y_out;
			}
			stripes_input_offset += stripe_input_block_size<<STRIPES_DIV;
			stripes_output_offset += stripe_output_block_size<<STRIPES_DIV;
    	}
    }
}
