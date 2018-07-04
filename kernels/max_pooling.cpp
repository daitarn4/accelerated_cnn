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
void hw_maxpooling_fpga_x2_striped(__global float *restrict inbuf, __global float *restrict outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   int output_filter_size,
								   int size_x)
#else
void hw_maxpooling_fpga_x2_striped(float* inbuf, float* outbuf,
								   int batch,
								   int in_f,
								   int filter_size,
								   int output_filter_size,
								   int size_x)
#endif
{
    // Use sliding window for max pooling layer
	float previous_line[1024];
	int output_index;
    for(int b = 0; b < batch; ++b){
    	for (int i_f = 0; i_f < in_f; i_f += STRIPES)
    	{
    		// Handle stripes accordingly
    		for (int s = 0; s < STRIPES; s++)
    		{
    			short x,y;
    			x = y = 0;
    			int c = s + ((i_f)*filter_size);
    			int d = s + (i_f*output_filter_size);
    			float val_previous=0;
    			float val = 0;
				#pragma ivdep
    			for (int i = 0; i < filter_size; i++)
    			{
    				int address = c;
    				val_previous = val;
    				val = inbuf[address];
    				if ((y&0x1)==0)
    					previous_line[x] = val;
    				else
    				{
    					if ((x&0x1) == 1)
    					{
    						// the four pixels to check are now available
    						// Find the maximum.
    						float vals[4];
    						vals[0] = previous_line[x-1];
    						vals[1] = previous_line[x];
    						vals[2] = val_previous;
    						vals[3] = val;
    						float max = vals[0];
#pragma unroll
    						for (int p = 1; p < 4; p++)
    							if (vals[p] > max) max = vals[p];
    						outbuf[d] = max;
    						d+=STRIPES;
    					}
    				}

    				x = x != (size_x-1)?x+1:0;
    				y = x == 0? y+1:y;
    				c+=STRIPES;
    			}
    		}
    	}
    }
}
