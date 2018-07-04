/*
 * short_cut_fpga.cpp
 *
 *  Created on: May 14, 2018
 *      Author: rchamberlain
 *
 *      Shortcut layer kernel. Simple, no optimisation required.
 *
 */



#include "shortcut_layer_fpga.h"

#ifdef ALTERA_CL
__kernel
void shortcut_layer_fpga(int input_block_size,
					     int batch, int w1, int h1, int c1,
						 __global volatile float *restrict add,
						 int w2, int h2, int c2, float s1, float s2,
						 __global volatile float *restrict in,
						 __global volatile float *restrict out,
						 __global volatile float *restrict out2 // Same as out, but allows pipelining!
						 )
#else
void shortcut_layer_fpga(int input_block_size,
				  int batch, int w1, int h1, int c1,
				  float *add, int w2, int h2, int c2, float s1, float s2,
				  float *in,float *out,float *out2)
#endif
{
	// First copy data to output buffer
	//for (int i = 0; i < input_block_size; i+=STRIPES)
	//		#pragma unroll	
          //      	for (int kk = 0; kk < STRIPES; kk++)
		//	out[i+kk] = in[i+kk];
    short stride = w1/w2;
    short sample = w2/w1;
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    short minw = (w1 < w2) ? w1 : w2;
    short minh = (h1 < h2) ? h1 : h2;
    short minc = (c1 < c2) ? c1 : c2;

    short i,j,k,b;
    for(b = 0; b < batch; ++b){
	k = 0;
	j = 0;
	i = 0;
	while (k < minc)
	{
        //for(k = 0; k < minc; k+= STRIPES){
         //   for(j = 0; j < minh; ++j){
          //      for(i= 0; i < minw; ++i){
			#pragma unroll	
                	for (int kk = 0; kk < STRIPES; kk++)
                	{
                		int kd = k >> STRIPES_DIV;
                		int out_index = kk + ((i*sample + w2*(j*sample + h2*(kd + c2*b)))<<STRIPES_DIV);
                		int add_index = kk + ((i*stride + w1*(j*stride + h1*(kd + c1*b)))<<STRIPES_DIV);
                		out2[out_index] = s1*in[out_index] + s2*add[add_index];
                	}
             //   }
           // }
        //}
	i = i != (minw-1)?i+1:0;
	j = i == 0 ? ((j != (minh - 1)) ? j + 1 : 0 ) : j;
	k = ((j == 0)&&(i==0)) ? k + STRIPES:k;
	}	
    }
}
