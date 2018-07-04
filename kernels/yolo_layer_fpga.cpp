/*
 * yolo_layer_fpga.cpp
 *
 *  Created on: May 15, 2018
 *      Author: rchamberlain
 */
#include "yolo_layer_fpga.h"

inline int entry_index_fpga(unsigned short w,
							unsigned short h,
							unsigned short classes,
							unsigned int location,
							int batch, int entry,
							int input_block_size)
{
    unsigned int n =   location / (w*h);
    unsigned int loc = location % (w*h);
    printf("block start of yolo layer = %d\n",((n*(4+classes+1))+entry));
    printf("loc start of yolo layer = %d\n",loc);
    return input_block_size + n*w*h*(4+classes+1) + entry*w*h + loc;
}


inline void activate_array_logistic_fpga(float *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
    	{
    		float lx = x[i];
    		lx = 1.0/(1.0 + exp(-lx));
    		x[i] = lx;
    	}
    }
}

#ifdef ALTERA_CL
__kernel
void yolo_layer_fpga(__global unsigned int *restrict layer_mask,
					 __global float *restrict in,
					 __global float *restrict out,
					 __global float *restrict out2,
					 int input_block_size,
					 short batch, short l_n,short classes,short w, short h,short c
					 )
#else
void yolo_layer_fpga(unsigned int *layer_mask,float *in, float *out, float  *out2, int input_block_size,
					 short batch, short l_n,short classes,short w, short h,short c)
#endif
{
	// Create a mask to only apply the
	bool active_layers[1024];
	for (int i = 0; i < 1024; i++)
		active_layers[i] = false;
	for (int i = 0; i < input_block_size; i++)
			out[i] = in[i];
	unsigned batch_offset = 0;
	short b,n;
	// Can do this on host and up load valid filters to modify. Can use mask to apply calculation to
	// striped input
	for (b = 0; b < batch; ++b){
			for (unsigned short i = 0; i < c; i+= STRIPES)
			{
				unsigned int bits = layer_mask[i/STRIPES];
				bits = (bits >> (i%STRIPES))&STRIPES_BIT_MASK;
				// Apply activation to these layers.
				for (int j = 0; j < w*h; j++)
				{
					for (int p = 0; p < STRIPES; p++)
					if ((bits >> p)&0x1)
					if ((i+p) < c)
					{
						int index = batch_offset + (j <<STRIPES_DIV)+ p + (i*w*h);
				   		float lx = out[index];
				    	lx = 1.0/(1.0 + exp(-lx));
				    	out2[index] = lx;
					}
				}
			}
            batch_offset += input_block_size;
	    }
}



