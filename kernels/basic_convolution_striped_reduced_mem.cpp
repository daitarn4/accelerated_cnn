/*
 * basic_convolution_striped.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 *
 */


#include "basic_convolution_striped_reduced_mem.h"
#include "fpga_channels.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable


/*
 * Coefficient kernel
 */

#ifdef ALTERA_CL
channel coeff_vector coeff_channel_rdmem __attribute__((depth(1024)));
#else
CHAN(coeff_vector,1024) coeff_channel_rdmem;

#endif



#ifdef ALTERA_CL
__kernel
#else
unsigned int count = 0;
extern "C"
#endif
void basic_convolution_striped_load_coeffs_kernel_reduced_mem(
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
					   int stride,
					   int batches_of_49,
					   int y_div)
{
	int ksize =  (kernel_size*kernel_size);
	char last = 0;
	int i,j,o_f,i_f,h,w,p,q;
	for (int y = 0; y < y_div; y++)
	{
		for( i = 0; i < batch; ++i)
		{
			for( j = 0; j < groups; ++j)
			{
				// Coefficient count
				unsigned int c=0;
				for ( o_f = 0; o_f < out_f; o_f +=STRIPES)
				{
					unsigned int out_batch = 0;
					unsigned int c_cached = c;
					for (out_batch = 0; out_batch < batches_of_49; out_batch++)
					{
						c = c_cached; // Repeat coefficients for multiple batches

						for ( i_f = 0; i_f < in_f; i_f += STRIPES)
						{

							for ( h = -pad; h < (kernel_size-pad); h++)
							{
								last = 
												(i_f == (in_f-STRIPES)) &&
												(h ==  (kernel_size-pad-1))? 1: 0;
								for ( w =-pad; w < (kernel_size-pad+last); w++)
								{
									for ( p = 0; p < STRIPES; p++)
									{
										coeff_vector vec;
		#pragma unroll
										for ( q = 0; q < STRIPES; q++)
										{
											vec.coeffs[q] = (w==(kernel_size-pad))?0:coeffs[c++];
										}
										write_channel_intel(coeff_channel_rdmem,vec);
	#ifndef ALTERA_CL
										count++;
	#endif
									}
								}
							}
						}
					}
				}
			}
		}
	}
}


#ifdef ALTERA_CL
__kernel
#else
//extern "C"
#endif
void basic_convolution_striped_kernel_reduced_mem(
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
					   int stride,
					   int batches_of_49,
					   int y_div) // Reduces the amount of output cahce memory required
{
	// coefficients are double buffered and read from parallel kernel
	// this allows time when memory is quiet to cache coefficients ahead of time
	// hopefully reduing any bottlenecks
	coeff_vector kernel_mask_A[STRIPES];
	coeff_vector kernel_mask_B[STRIPES];
#define OUTPUT_CACHE_SIZE (112*112)
#ifndef ALTERA_CL
	auto img_cache = new float[ 4*MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES];
	auto out_cache = new float[ MAX_BATCH_SIZE][STRIPES];
#else
	float img_cache[2*MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES];
	float out_cache[MAX_BATCH_SIZE][STRIPES];
#endif

	short ksize =  (kernel_size*kernel_size);
	short sizex,sizey;
	short out_sizex,out_sizey;
	sizex = size;
	sizey = (short)size/(short)y_div; // Move to host
	out_sizex = out_size;
	out_sizey = (short)out_size/(short)y_div;
	short y_div_offset = 0;
	bool toggle = 0;
	unsigned int out_y_div_offset=0;
	char pad_compensate = pad;
	unsigned int offset_increment = (sizex*(sizey+(pad_compensate<<1)));
	for (int yy_div = 0; yy_div < y_div; yy_div++)
	{
		y_div_offset = yy_div*sizey;
		unsigned int c,d;
		c=d=0;
		{
			// Add vertical padding
			for (short i_f = 0; i_f  < in_f; i_f+=STRIPES)
			{
				short x,y;
				x = 0;
				y = -pad_compensate + (sizey*yy_div);
				int input_index = (i_f*size*size)+  (yy_div*sizey*sizex<<STRIPES_DIV);
				int s = (i_f*size*size) + ((yy_div*sizey-pad_compensate)*(sizex<<STRIPES_DIV));
				while (y < ((1+yy_div)*sizey+pad_compensate))
				{
					int input_index2 = s;
					// check for halo
					bool halo = ((x < 0) || (x >= size) || (y >= size) || (y < 0))?true:false;

					#pragma unroll
					for (int p = 0; p < STRIPES; p++)
					{
						float val = halo?0:(input_index2 < 0?0:input[input_index2+p]);

						img_cache[d][p] = val;
					}
					x = x != (sizex-1)? x+1:0;
					y = x == 0 ? y+1:y;
					s+=STRIPES;
					d++;
				}
			}
		}
		// Coefficient count
		c = 0;
		unsigned int o=0;
		unsigned int output_offset = 0;

		for (short o_f = 0; o_f < out_f; o_f +=STRIPES)
		{
			unsigned int coefficent_load_count = 0;
			unsigned int batch_pointer = 0;
			unsigned int out_batch_offset = 0;
			for (short out_batch = 0; out_batch < batches_of_49; out_batch++)
			{
				short x,y;
				short xx,yy;
				unsigned int in_offset = 0;//(j*in_f*sizex*sizey) + (i*groups*in_f*sizex*sizey);
				short out_batch_count = 0;
				short i_f = 0;
				char first = 1;
				char w = -(pad+first);
				char h = -(pad);
				xx = batch_pointer%(sizex);
				yy = (batch_pointer/(sizex))*stride;
				x = xx;
				y = yy;


				#pragma ivdep
				while (i_f < in_f)
				{
					if (coefficent_load_count < STRIPES)
					{
						coeff_vector vec = read_channel_intel(coeff_channel_rdmem);
#ifndef ALTERA_CL
								count--;
#endif

						if (toggle)
						{
							#pragma unroll
							for (int p = 0; p < (STRIPES-1);p++)
							{
								kernel_mask_B[p] = kernel_mask_B[p+1];
							}
							kernel_mask_B[(STRIPES-1)] = vec;
						}
						else
						{
							#pragma unroll
							for (int p = 0; p < (STRIPES-1);p++)
							{
								kernel_mask_A[p] = kernel_mask_A[p+1];
							}
							kernel_mask_A[(STRIPES-1)] = vec;
						}
					}

					bool out_of_bounds = ((h+y+y_div_offset) < 0) || ((w+x) < 0) || ((h+y+y_div_offset) >= size) || ((w+x) >= size);
					int index = (w+x) + ((h+(y+pad_compensate))*sizex) + in_offset; // Compensate for extra padding held in memory

#pragma unroll
					for (int q = 0; q < STRIPES; q++)
					{
						float res = 0;
						if (!out_of_bounds)
						{
							#pragma unroll
							for (int p = 0; p < STRIPES; p++)
							{
	#ifndef ALTERA_CL
								float value = index >= 0?img_cache[index][p]:0;
	#else

								float value =index >= 0?img_cache[index][p]:0;
	#endif

								//res += kernel_mask_A[p+(q*STRIPES)]*value;
								res += (toggle?(kernel_mask_A[q].coeffs[p]):kernel_mask_B[q].coeffs[p])*value;
							}
						}
						if (((i_f == 0) && (h<=-pad) && (w ==-pad)))
							out_cache[out_batch_count][q] = res;
						else
							out_cache[out_batch_count][q] +=res;

					}


					// Calculate all indexes
					out_batch_count = out_batch_count != (MAX_BATCH_SIZE-1)?out_batch_count+1:0;

					if ((out_batch_count==0) && (h == (kernel_size-pad-1)) && (w == (kernel_size-pad-1)))
					{
						in_offset+= offset_increment;// (sizex*(sizey+(pad_compensate<<1)));
						i_f+= STRIPES;
					}

					x = (out_batch_count==0) ? xx : (x != (sizex-stride)? x+stride:0);
					y =  (out_batch_count==0) ? yy : ((x == 0)? (y != ((sizey+pad_compensate+pad_compensate)-stride)?y+stride:0):y);

					coefficent_load_count = (out_batch_count==0)? 0 : coefficent_load_count+1;

					toggle =(out_batch_count==0)?(toggle?0:1):toggle;
					w =(out_batch_count==0)? ((w != (kernel_size-pad-1))?w+1:-(pad+first)):w;
					h =((out_batch_count==0) && (w==-(pad+first)))?((h != (kernel_size-pad-1))?h+1:-(pad)):h;
					first =(out_batch_count==0) ? 0 : first;

				}
				unsigned short cache=0;
				//unsigned int oo = (o_f *out_size*out_size) + (STRIPES*out_batch*MAX_BATCH_SIZE) + (STRIPES*yy_div*out_sizey*out_sizex);
				unsigned int oo = output_offset + out_batch_offset + out_y_div_offset;
				unsigned int oo_x,oo_y;

				for ( cache = 0; cache < MAX_BATCH_SIZE; cache++)
				{

					#pragma unroll
					for (int q = 0; q < STRIPES; q++)
					{
						output[oo++] = out_cache[cache][q];
					}
				}
				batch_pointer += (MAX_BATCH_SIZE*stride);
				out_batch_offset += (STRIPES*MAX_BATCH_SIZE);
			}
			output_offset += (STRIPES*out_size*out_size); // Group of stripes targeted
		}
		out_y_div_offset += (STRIPES*out_sizey*out_size); // offset within each striped block
	}
#ifndef ALTERA_CL
	printf("count = %d\n",count);
	delete img_cache;
	delete out_cache;

#endif
}

