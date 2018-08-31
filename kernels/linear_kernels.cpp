/*
 * linear_kernels.cpp
 *
 *  Created on: May 24, 2018
 *      Author: rchamberlain
 *
 *  This kernel supports all linear kernels to reduce replication of global memory accesses
 *
 *
 */
#include "basic_convolution_striped.h"

#define DTYPE float
enum {LEAKY_ACTIVATION=1,
	  MAX_POOLING_x2=2};


#ifdef ALTERA_CL
__kernel
void linear_kernels(unsigned char KERNEL_TYPE,
					__global DTYPE *restrict input0,
					__global DTYPE *restrict input1,
					__global DTYPE *restrict output0,
					// Memory offsets
					unsigned int offset0,
					unsigned int offset1,
					unsigned int offset2,
					unsigned int offset3,
					unsigned int offset4,
					DTYPE fconstant0,
					DTYPE fconstant1,
					int iconstant0,
					int iconstant1,
					int iconstant2,
					unsigned short l_c,
					unsigned short l_n,
					unsigned short w_out,
					unsigned int batch,
					unsigned int size,
					unsigned int feature_loop_size) // The size of loop to iterate over
#else
void linear_kernels(unsigned char KERNEL_TYPE,
					DTYPE *input0,
					DTYPE *input1,
					DTYPE *output0,
					// Memory offsets
					unsigned int offset0,
					unsigned int offset1,
					unsigned int offset2,
					unsigned int offset3,
					unsigned int offset4,
					DTYPE fconstant0,
					DTYPE fconstant1,
					int iconstant0,
					int iconstant1,
					int iconstant2,
					unsigned short l_c,
					unsigned short l_n,
					unsigned short w_out,
					unsigned int batch,
					unsigned int size,
					unsigned int feature_loop_size) // The size of loop to iterate over
#endif
{

	unsigned int batch_offset = 0;

	// Preamble. Some layers will require setup.
	// Leaky activation
	#define MAX_FEATURES 2048
	#define MAX_FEATURE_SIZE 1024
	DTYPE div_sqrt_variance[MAX_FEATURES];
	DTYPE rolling_mean[MAX_FEATURES];
	DTYPE scales[MAX_FEATURES];
	DTYPE biases[MAX_FEATURES];
	// Max Pool
	DTYPE previous_line[MAX_FEATURE_SIZE*100][STRIPES];

	switch (KERNEL_TYPE)
	{
	 	 case LEAKY_ACTIVATION :
	 	 {
	 		 unsigned short c = 0;
	 		 unsigned short d = 0;
	 		 unsigned int d_offset = 0;
	 		 while (d < 4)
	 		 {
	 			unsigned int index = d_offset+c;
	 			DTYPE in[STRIPES];
#pragma unroll
	 			for (int p = 0; p < STRIPES; p++)
	 				in[p] = input1[index + p];
	 			if (d == 0)
					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
						div_sqrt_variance[c] = in[p];
	 			if (d == 1)
					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
						rolling_mean[c] = in[p];
	 			if (d == 2)
					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
						scales[c] = in[p];
	 			if (d == 3)
					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
						biases[c] = in[p];
				c = c != (l_c-STRIPES)?c+STRIPES:0;
	 			d_offset = c == 0? d_offset+l_c : d_offset;
	 			d = c == 0? d + 1: d;
	 		 }
	 	 }
	 	 break;
	 	 default :
	 		 // No pre-amble
	 	 break;
	}


	for(unsigned int b = 0; b < batch; ++b) // Batches
	{
		 unsigned int loop_offset = 0;
		 for (int i = 0; i < feature_loop_size; i+=STRIPES) // Features
		 {
			 // x and y coordinates
			 unsigned short x,y;
			 DTYPE previous[STRIPES];
			 x = y = 0;
			 #pragma ivdep
			 for (int j = 0; j < size; j++) // pixels
			 {
				 unsigned int index = (batch_offset + loop_offset + j) << STRIPES_DIV;
				 unsigned int output_index;
				 DTYPE input[STRIPES];
				 DTYPE output[STRIPES];
				 // For max pool
			 	 #pragma unroll
				 for (int p = 0; p < STRIPES; p++)
					previous[p] = input[p];

				 #pragma unroll
				 for (int p = 0; p < STRIPES; p++)
					 input[p] = input0[p + index];
				 bool write_output_valid = false;
				 switch (KERNEL_TYPE)
				 {
				 	case LEAKY_ACTIVATION :
				 		output_index = index;
					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
					{
						float norm;
						float bias;
						float leaky_activation;
						float scale;

						if (iconstant0 == 1)
						{
							norm =  (input[p] - rolling_mean[i+p])*div_sqrt_variance[i+p];
							scale = scales[i+p];
						}
						else
						{
							norm = input[p];
							scale =1.0f;
						}
						bias = norm*scale + biases[i+p];
						leaky_activation = bias < 0.0f? 0.1f*bias:bias;
						output[p] = leaky_activation;
						write_output_valid = true;
					}
					break;
					case MAX_POOLING_x2:

					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
					{
						if ((y&0x1)==0)
							previous_line[x][p] = input[p];
						else
						{
	    					if ((x&0x1) == 1)
	    					{
	    						// the four pixels to check are now available
	    						// Find the maximum.
	    						DTYPE vals[4];
	    						vals[0] = previous_line[x-1][p];
	    						vals[1] = previous_line[x][p];
	    						vals[2] = previous[p];
	    						vals[3] = input[p];
	    						DTYPE max = vals[0];
	    						write_output_valid = true;
	#pragma unroll
	    						for (int q = 1; q < 4; q++)
	    							if (vals[q] > max) max = vals[q];
	    						output[p] = max;
	    					}
						}
					}
					break;
					default : break;
				 }

				 if (write_output_valid)
	#pragma unroll
					 for (int p = 0; p < STRIPES; p++)
						 	output0[output_index + p] = output[p];
				 x = x != (w_out-1)? x+1 : 0;
				 y = x == 0? y + 1: y;
			 }
		  }
	 }
}



