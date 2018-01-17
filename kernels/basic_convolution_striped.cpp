/*
 * basic_convolution_striped.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 *
 */


#include "basic_convolution_striped.h"




#pragma OPENCL EXTENSION cl_intel_channels : enable



#ifndef ALTERA_CL
/*
 * Helper functions to stripe input data and coefficients
 */

/*
 * Function to re-order coefficients to match striping in convolution function
 * Data is striped evenly on input and output
 */
  extern "C" void stripe_coefficients(int in_f,int out_f, int kernel_size,float *in,float *out)
{
	int size = (kernel_size*kernel_size);
	int out_index=0;
	int i,j,k,q,p;

	for (i = 0; i < out_f; i+= STRIPES)
	{
		for ( j = 0; j < in_f; j+= STRIPES)
		{
			for ( k = 0; k < size;k++)
			{
				for ( q = 0; q < STRIPES; q++)
				{
					for ( p = 0; p < STRIPES; p++)
					{
						int index_in = (j+p)*size; // Offset in input
							index_in += (q+i)*in_f*size; // Offset in output
							index_in += k; // Offset in kernel
						float val;
						if (((j+p) >= in_f) || ((i+q) >= out_f))
							val = 0;
						else
							val = in[index_in];
						out[out_index++] = val;
					}
				}
			}
		}
	}
}

/*
 * Stripe input data
 */
  extern "C" void stripe_input_data(int in_f,int size,float *in,float *out)
{
	int i,j,k;
	unsigned int c= 0;
	for ( j = 0; j < in_f; j+=STRIPES)
	for ( i = 0; i < size; i++)
	{
		for ( k = 0; k < STRIPES; k++)
		{
			float val;
			if ((j+k) >= in_f)
				val = 0;
			else
				val = in[(size*(j+k))+i];
			out[(size*j)+(i<<STRIPES_DIV)+k] = val;
		}
	}
}

/*
 * Remove striping output data
 */
  extern "C" void remove_stripes(int in_f,int size,float *in,float *out)
{
	unsigned int c=0;
	int j,i,k;
	for ( j = 0; j < in_f; j+=STRIPES)
	for ( i = 0; i < size; i++)
	{
		for ( k = 0; k < STRIPES; k++)
		{
			if ((j+k) < in_f)
				out[(size*(j+k))+i] = in[c];
			c++;
		}
	}
}
#endif


/*
 * Coefficient kernel
 */


#ifdef ALTERA_CL
channel coeff_vector coeff_channel __attribute__((depth(1024)));
#else
coeff_vector *coeff_channel = NULL;
unsigned int coeff_input_counter = 0;
unsigned int coeff_output_counter = 0;
void write_channel_intel(coeff_vector *chan,coeff_vector value)
{
	if (!coeff_channel)
	{
		coeff_channel = (coeff_vector*)malloc(sizeof(coeff_vector)*1024*1024*9); // Big array to avoid over flowing in simulation of channels
	}
	coeff_channel[coeff_input_counter++] = value;
}

coeff_vector read_channel_intel(coeff_vector *chan)
{
	return chan[coeff_output_counter++];
}

extern "C" void clear_channel(){free(coeff_channel);coeff_channel=NULL;coeff_input_counter=0;coeff_output_counter=0;}

#endif

#ifdef ALTERA_CL
__kernel
#endif
void basic_convolution_striped_load_coeffs_kernel(
#ifdef ALTERA_CL
					   __global float *restrict coeffs,
#else
					   float *coeffs,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int out_f,
					   int stride)
{
	int ksize =  (kernel_size*kernel_size);
	char first = 1;
	int i,j,o_f,i_f,h,w,p,q;
    for( i = 0; i < batch; ++i)
	{
        for( j = 0; j < groups; ++j)
		{
			// Coefficient count
			unsigned int c=0;
			for ( o_f = 0; o_f < out_f; o_f +=STRIPES)
			{
				for ( i_f = 0; i_f < in_f; i_f += STRIPES)
				{
					for ( h = -pad; h < (kernel_size-pad); h++)
					{
						for ( w =-(pad+first); w < (kernel_size-pad); w++)
						{
							for ( p = 0; p < STRIPES; p++)
							{
								coeff_vector vec;
#pragma unroll
								for ( q = 0; q < STRIPES; q++)
								{
									vec.coeffs[q] = coeffs[c++];
								}
								write_channel_intel(coeff_channel,vec);
							}
						}
						first = 0; // Extra batch for final coeffs that are not used
					}
				}
			}
        }
	}
}


//#define PRUNE
//#define COUNT_ZEROS
//#define THRESHOLD 0.004f

#ifdef ALTERA_CL
__kernel
#endif
void basic_convolution_striped_kernel(
#ifdef ALTERA_CL
					   __global float *restrict input,
					   __global float *restrict coeffs,
					   __global float *restrict output,
#else
					   float *input, float *coeffs, float *output,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int out_f,
					   int stride)
{
	// coefficients are double buffered and read from parallel kernel
	// this allows time when memory is quiet to cache coefficients ahead of time
	// hopefully reduing any bottlenecks
	coeff_vector kernel_mask_A[STRIPES];
	coeff_vector kernel_mask_B[STRIPES];
#define OUTPUT_CACHE_SIZE (112*112)
	#ifndef ALTERA_CL
	float **img_cache = new float*[STRIPES];
	for (int i = 0; i < STRIPES; i++)
		img_cache[i] = new float[(401408/STRIPES)];

	float **out_cache = new float*[STRIPES];
	for (int i = 0; i < STRIPES; i++)
		out_cache[i] = new float[OUTPUT_CACHE_SIZE];

	//float (*img_cache)[STRIPES] = (float*)malloc(sizeof(float[(401408/STRIPES)][STRIPES]));
	//float (*out_cache)[STRIPES] = (float*)malloc(sizeof(float[OUTPUT_CACHE_SIZE][STRIPES]));

	#else
	float img_cache[401408/STRIPES][STRIPES];
	float out_cache[OUTPUT_CACHE_SIZE][STRIPES];
	#endif
#ifdef COUNT_ZEROS
unsigned int zeros = 0;
unsigned int zero_coeffs = 0;
#endif

	int ksize =  (kernel_size*kernel_size);
	char first = 1;
	int i,h,s,i_f,o_f,x,y,j,p,q;
    for( i = 0; i < batch; ++i)
	{
        for( j = 0; j < groups; ++j)
		{
#define STRIPED_INPUT
#ifdef STRIPED_INPUT
        	unsigned int c,d;
        	c=d=0;
        	while (d < (in_f*size*size>>STRIPES_DIV))
        	{
#pragma unroll
				for ( s = 0; s < STRIPES; s++)
				{
					img_cache[d][s] = input[c++];
				}
				d++;

        	}
#else
			for ( i_f = 0; i_f < in_f; i_f+=STRIPES)
			{
				unsigned int in_offset = (i_f*size*size) + (j*in_f*size*size) + (i*groups*in_f*size*size);
				for ( y = 0; y < size; y++)
				{
					for ( x = 0; x < size; x++)
					{
						int index = x + (y*size) + in_offset;
						//index = STRIPE_INDEX(PAR_INPUTS,i_f,)
						for ( s = 0; s < STRIPES; s++)
						{
							img_cache[x + (y*size)+(in_offset>>STRIPES_DIV)][s] = input[x + (y*size) + in_offset +  (s*size*size)];
						}
					}
				}
			}
#endif
			// Coefficient count
			c = 0;
			unsigned int o=0;
			char toggle = 0;


			for ( o_f = 0; o_f < out_f; o_f +=STRIPES)
			{
				unsigned int in_offset = (j*in_f*size*size) + (i*groups*in_f*size*size);
				char w = -(pad+first);
				char h = -(pad);
				short x,y;
				x=y=0;
				unsigned short coefficent_load_count = 0;
				#pragma ivdep
				short i_f = 0;
				while (i_f < in_f)
				{
					if (coefficent_load_count < STRIPES)
					{
						coeff_vector vec = read_channel_intel(coeff_channel);
						if (toggle)
						{
							#pragma unroll
							for ( p = 0; p < (STRIPES-1);p++)
							{
								kernel_mask_B[p] = kernel_mask_B[p+1];
							}
							kernel_mask_B[(STRIPES-1)] = vec;
						}
						else
						{
							#pragma unroll
							for ( p = 0; p < (STRIPES-1);p++)
							{
								kernel_mask_A[p] = kernel_mask_A[p+1];
							}
							kernel_mask_A[(STRIPES-1)] = vec;
						}
					}

					int out_of_bounds = ((h+y) < 0) || ((w+x) < 0) || ((h+y) >= size) || ((w+x) >= size);
					int index = (w+x) + ((h+y)*size) + (in_offset>>STRIPES_DIV);
					const short cache_index = ((x/stride) + ((y/stride) * out_size));
#pragma unroll
					for ( q = 0; q < STRIPES; q++)
					{
						float res = 0;
						if (!out_of_bounds)
						{
							#pragma unroll
							for ( p = 0; p < STRIPES; p++)
							{
								float value = img_cache[index][p];
#ifdef COUNT_ZEROS
								if ((o_f == 0) && (w==0) && (h==0) &&!out_of_bounds && (fabs(value) < THRESHOLD))
								{
									zeros++;
								}
								if ((x == 0) && (y==0)&&(fabs(kernel_mask[p+(q*STRIPES)]) < THRESHOLD))
									zero_coeffs++;
#endif

								//res += kernel_mask_A[p+(q*STRIPES)]*value;
								res += (toggle?(kernel_mask_A[q].coeffs[p]):kernel_mask_B[q].coeffs[p])*value;
							}
						}
						if ((i_f == 0) && (h<=-pad) && (w ==-pad))
							out_cache[cache_index][q] = res;
						else
							out_cache[cache_index][q] += res;
					}
					// Handle indexes! Deep combinatorial, probably will cause low fmax.
					x = x != (size-stride)? x+stride:0;
					y = (x == 0)? (y != (size-stride)?y+stride:0):y;
					coefficent_load_count = ((y == 0) && (x == 0)) ? 0 : coefficent_load_count+1;
					if (coefficent_load_count == 0)
						printf(".");
					toggle = ((y == 0) && (x == 0))?(toggle?0:1):toggle;

					if ((coefficent_load_count==0) && (h == (kernel_size-pad-1)) && (w == (kernel_size-pad-1)))
					{
						in_offset+= (STRIPES*size*size);
						i_f+= STRIPES;
					}

					w =(coefficent_load_count==0)? ((w != (kernel_size-pad-1))?w+1:-(pad+first)):w;
					h =((coefficent_load_count==0) && (w==-(pad+first)))?((h != (kernel_size-pad-1))?h+1:-(pad)):h;
					first = ((y == 0) && (x == 0)) ? 0 : first;
				}
				unsigned int cache=0;
				for ( y = 0; y < size; y+=stride)
				{
					for ( x = 0; x < size; x+=stride)
					{
					    #pragma unroll
						for ( q = 0; q < STRIPES; q++)
						{
							output[o++] = out_cache[cache][q];
						}
						cache++;
					}
				}
			}
        }
	}
	#ifndef ALTERA_CL
	free(img_cache);
	free(out_cache);
	#endif
#ifdef COUNT_ZEROS
	printf("%f%%\t",(100.0f*(float)zeros/(float)(in_f*size*size)));
	printf("%f%%\n",(100.0f*(float)zero_coeffs/(float)(in_f*out_f*kernel_size*kernel_size)));
#endif

}

