/*
 * basic_convolution_striped.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 *
 */


#include "conv_binary_fpga_v4.h"
#include "fpga_channels.h"
#include "bnn_libraries.h"
#ifndef ALTERA_CL
#include <stdlib.h>
#endif

#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct {int s[STRIPES];} vec_int;
typedef struct {unsigned int s[STRIPES];} vec_uint;
typedef struct {short s[STRIPES];} vec_short;


/*
 * Coefficient kernel
 */

#ifdef ALTERA_CL
channel vec_uint coeff_channel_rdmem_binary2 __attribute__((depth(8)));

#else
CHAN(vec_uint,8) coeff_channel_rdmem_binary2;

#endif


#ifdef ALTERA_CL
channel int input_channels[STRIPES] __attribute__((depth(8)));

#else
CHAN(int,8) input_channels[STRIPES];

#endif

#ifdef ALTERA_CL
__attribute__((task()))
__kernel
#else
extern "C"
#endif
void conv_coeffs_binary_subblock_fpga(
#ifdef ALTERA_CL
					   __global unsigned int *restrict coeffs,
#else
					   unsigned int *coeffs,
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
					   int y_div,
					   int block_size)
{
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
					//for (out_batch = 0; out_batch < batches_of_49; out_batch++)
					{
						c = c_cached; // Repeat coefficients for multiple batches
						int ncoeffs = block_size;//+STRIPES;//(in_f*kernel_size*kernel_size+STRIPES);
						for (int i = 0; i < ncoeffs;i+=STRIPES)
						{
							vec_uint v;
#pragma unroll STRIPES
							for (q = 0; q < STRIPES; q++)
							{
								unsigned int vec = coeffs[c+q];
								v.s[q] = vec;
							}
							write_channel_intel(coeff_channel_rdmem_binary2,v);
							//if (i < (ncoeffs-STRIPES))
								c+=STRIPES;
						}
					}
				}
			}
		}
	}
}




#ifndef ALTERA_CL
int out_count = 0;

float max_val = 0.0f;
float min_val = 0.0f;
#endif


inline vec_int AccumulateStripes(vec_uint weights,vec_short input,vec_int out,
					          unsigned short i_f,unsigned short l_c)
{
	vec_int sum;
#pragma unroll STRIPES
	for (int q = 0; q < STRIPES; q++)
	{
		int res = 0;
		unsigned int bits = weights.s[q];
		#pragma unroll STRIPES
		for (int p = 0; p < STRIPES; p++)
		{
			short value = input.s[p];
			int add_sub2 = GetBit2(bits,p);
			if ((i_f + p ) < l_c)
				res += (add_sub2&0x1)==1?value:-value;
		}
		sum.s[q] = out.s[q] + res;
	}
	return sum;
}


#ifdef ALTERA_CL
__attribute__((task()))
__kernel
#else
//extern "C"
#endif
void conv_binary_subblock_fpga_v4(
#ifdef ALTERA_CL
					   __global volatile float *restrict input,
					   __global volatile unsigned int *restrict coeffs,
					   __global volatile float *restrict output,
#else
					   float *input, unsigned int *coeffs, float *output,
#endif
					   unsigned short no_sub_blocks_y,
					   unsigned short no_sub_blocks_x,
					   int sub_block_width,
					   int sub_block_height,

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
					   int y_div,
#ifdef ALTERA_CL
					   __global float *restrict binary_scale,
#else
					   float *binary_scale,
#endif
					   int batch_size,
					   int step_size,
					   int out_step_size,
					   int l_c,
					   unsigned int total_block_count
					  ) // Reduces the amount of output cahce memory required
{
	// coefficients are double buffered and read from parallel kernel
	// this allows time when memory is quiet to cache coefficients ahead of time
	// hopefully reducing any bottlenecks

	// No longer splitting in Y as this is probablematic for large images with odd
	// striding

	// Now splitting image into blocks with halo big enough of stride and filter size

	// |-----|-----|
	// |     |     |
	// |  1  |  2  |
	// |-----|-----|
	// |     |     |
	// |  3  |  4  |
	// |-----|-----|


	// Loop of the number of sub blocks

	unsigned short sub_block_x,sub_block_y;
	sub_block_x = sub_block_y = 0;

	unsigned int sub_block_x_index = 0;
	unsigned int sub_block_y_index = 0;

	vec_uint kernel_mask_A;

	short index_offsets[9];
	char c = 0;
	for (char h = 0; h <= (pad<<1); h++)
		for (char w = 0; w <= (pad<<1); w++)
		{
			index_offsets[c++] = w + (h*(sub_block_width+(pad<<1)));
		}


#ifdef ALTERA_CL
	vec_short img_cache[MAX_INPUT_IMAGE_BATCH];
	vec_int out_cache[ MAX_OUTPUT_IMAGE_BATCH];
#else
	auto img_cache = new 	vec_short[MAX_INPUT_IMAGE_BATCH];
	auto out_cache = new 	vec_int[ MAX_OUTPUT_IMAGE_BATCH];
#endif


	bool toggle = false; // Toggle used select kernel buffer

	while (sub_block_y != no_sub_blocks_y)
	{
		// Temporary input buffer. Converted to short of binary addition calculation
		// Each block needs contain all input features
		//int out_cache[104*104][STRIPES];

		// Load in required data
		unsigned short sub_block_x_index = (sub_block_x * sub_block_width);
		unsigned short sub_block_y_index = (sub_block_y * sub_block_height);

		int pad = ((kernel_size-1)>>1);
		int pad2 = pad<<1;
		int copy_width = sub_block_width + pad2;
		int copy_height = sub_block_height + pad2;
		unsigned int c = 0;

		int i_f;
		short x,y;
		x = y =  -pad;
		i_f = 0;
		while (i_f < in_f)
		//for (int i_f = 0; i_f < in_f; i_f+=STRIPES)
		{
			unsigned int feature_offset = i_f * size * size;
			//for (int y = -pad; y < (sub_block_height+pad); y++)
			//{
			//	for (int x= -pad; x < (sub_block_width+pad); x++)
			//	{
					int index_x = x + sub_block_x_index;
					int index_y = y + sub_block_y_index;
					bool out_of_bounds = false;
					if ((index_x < 0) || (index_y < 0) || (index_x >= size) || (index_y >= size))
						out_of_bounds = true;
					int index = feature_offset + ((index_x) + (index_y * size))*STRIPES;

#pragma unroll STRIPES
					for (int p = 0; p < STRIPES; p++)
						img_cache[c].s[p] = out_of_bounds ? 0 : FloatToShort(input[index+p],BINARY_FLOAT_SCALE);
					c++;
			//	}
			//	}
			x = (x != (sub_block_width+pad-1))?x+1:-pad;
			y = (x == -pad) ? ((y != (sub_block_height+pad-1))?y+1:-pad):y;
			i_f = ((x==-pad) && (y == -pad))?i_f+STRIPES:i_f;
		}

		// Input is now cached. No loop over all possible outputs for this sub block
		// Variable to count in coefficients
		unsigned int coefficent_load_count = 0;
		// Variables for locating position in input filter
		short x2;
		x = 0; y = 0;
		unsigned char stride_shift = ((stride==2)?1:0);

		unsigned short block_size = (sub_block_width>>stride_shift) * ( sub_block_height>>stride_shift);
		unsigned short block_chunk = (copy_width * copy_height);
		for (short o_f = 0; o_f < out_f; o_f +=STRIPES)
		{
			char first = 0;
			char w = -first;
			char h = 0;

			short i_f = 0;
			unsigned short feature_offset = 0;
			unsigned short y_offset = 0;
			unsigned short feature_index = 0;
			x=y=0;

#define PAR_PIX 1
#pragma ivdep array(out_cache)
			while (i_f < in_f)
			{
				x2 = x + stride;
				// Load coefficients in parallel using double buffer
				if (feature_index == 0)
				{
						kernel_mask_A  = read_channel_intel(coeff_channel_rdmem_binary2);
				}
				unsigned short xx = x + w;
				unsigned short yy = y + h;

				unsigned short index = (xx + (yy*copy_width)) + feature_offset;
				unsigned short out_index = ((x>>stride_shift) + ((y>>stride_shift)*(sub_block_width>>stride_shift)));
				vec_short input_vals;

#pragma unroll STRIPES
				for (int p = 0; p < STRIPES; p++) input_vals.s[p] = ((i_f + p ) < l_c)?img_cache[index].s[p]:0;

				vec_int out,in,zero;
				vec_int out2,in2;
				#pragma unroll
				for (int p = 0; p < STRIPES; p++)
					zero.s[p] = 0;
				if ((h<=0) && (w <=0) && (i_f == 0)) in = zero; else in = out_cache[out_index];

				out = AccumulateStripes(kernel_mask_A,input_vals,in,i_f,l_c);

				out_cache[out_index] = out;		


				/*if ((h == (kernel_size-1)) && (w == (kernel_size-1)) && (i_f >= (in_f-STRIPES)))
				{
#pragma unroll STRIPES
					for (int p = 0; p < STRIPES; p++)
					{
						write_channel_intel(input_channels[p],out_cache[out_index].s[p]);
					}
				}*/

				feature_index = feature_index == (block_size-1)?0:feature_index + 1;
				// Loop of sizes this is always the case
				x = x != (sub_block_width - (stride*PAR_PIX)) ? x + stride : 0;
				y = (x == 0) ? ((y < (sub_block_height -stride)) ? y + stride : 0) : y;
				i_f = ((feature_index == 0) && ((h == (kernel_size-1)) && (w == (kernel_size-1))))?i_f+=STRIPES:i_f;
				feature_offset = ((feature_index == 0) && ((h == (kernel_size-1)) && (w == (kernel_size-1))))?feature_offset+block_chunk:feature_offset;

				w = (feature_index == 0) ? ((w != (kernel_size-1)) ? w+1 : 0) : w;
				h = (feature_index == 0) ? (w ==-(first))?((h != (kernel_size-1))?h+1:0):h : h;

				//first = ((x == 0) && (y == 0))?0:first;
				//coefficent_load_count = ((x == 0) && (y == 0))?0:coefficent_load_count+1;
				//toggle = ((x == 0) && (y == 0))?(toggle?false:true):toggle;

			}
			for (int i = 0; i < block_size;i++)
			{
#pragma unroll STRIPES
					for (int p = 0; p < STRIPES; p++)
					{
						write_channel_intel(input_channels[p],out_cache[i].s[p]);
					}
			}
		/*	x=y=0;
			int c_out = 0;
			while (y < sub_block_height)
			{

	#pragma unroll STRIPES
			    for (int p = 0; p < STRIPES; p++)
				{
						write_channel_intel(input_channels[p],out_cache[c_out].s[p]);
				}
				x = x != (sub_block_width-stride)?x+stride:0;
				y = x == 0 ? y+stride : y;
				c_out++;
			}*/

		}

		sub_block_x = sub_block_x != (no_sub_blocks_x - 1) ? sub_block_x + 1 : 0;
		sub_block_y = sub_block_x == 0 ? sub_block_y+1 : sub_block_y;
	}
#ifndef ALTERA_CL
		delete out_cache;
		delete img_cache;
#endif
}



// Output kernel
#ifdef ALTERA_CL
__attribute__((task()))
__kernel
#else
//extern "C"
#endif
void conv_activations_v4(
#ifdef ALTERA_CL
					   __global volatile float *restrict output,
#else
					   float *output,
#endif
					   unsigned short no_sub_blocks_y,
					   unsigned short no_sub_blocks_x,
					   int sub_block_width,
					   int sub_block_height,

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
					   int y_div,
#ifdef ALTERA_CL
					   __global float *restrict binary_scale,
#else
					   float *binary_scale,
#endif
					   int batch_size,
					   int step_size,
					   int out_step_size,
					   int l_c,
					   // adding activation to convolution
#ifdef ALTERA_CL
					  __global volatile float *restrict data,
#else
					  float *data,
#endif
					  int batch_normalised,
					  int activation

					  ) // Reduces the amount of output cahce memory required
{
	// coefficients are double buffered and read from parallel kernel
	// this allows time when memory is quiet to cache coefficients ahead of time
	// hopefully reducing any bottlenecks

	// No longer splitting in Y as this is probablematic for large images with odd
	// striding

	// Now splitting image into blocks with halo big enough of stride and filter size

	// |-----|-----|
	// |     |     |
	// |  1  |  2  |
	// |-----|-----|
	// |     |     |
	// |  3  |  4  |
	// |-----|-----|


	// Loop of the number of sub blocks

	unsigned short sub_block_x,sub_block_y;
	sub_block_x = sub_block_y = 0;

	unsigned int sub_block_x_index = 0;
	unsigned int sub_block_y_index = 0;

	bool toggle = false; // Toggle used select kernel buffer

	int total_reads = 0;

	while (sub_block_y != no_sub_blocks_y)
	{
		// Temporary input buffer. Converted to short of binary addition calculation
		// Each block needs contain all input features
		//int out_cache[104*104][STRIPES];

		// Load in required data
		unsigned short sub_block_x_index = (sub_block_x * sub_block_width);
		unsigned short sub_block_y_index = (sub_block_y * sub_block_height);

		int pad = ((kernel_size-1)>>1);
		int pad2 = pad<<1;
		int copy_width = sub_block_width + pad2;
		int copy_height = sub_block_height + pad2;
		unsigned int c = 0;

		int i_f;
		// Input is now cached. No loop over all possible outputs for this sub block
		// Variable to count in coefficients



		for (short o_f = 0; o_f < out_f; o_f +=STRIPES)
		{
			char first = 1;
			char w =-first;
			char h = 0;

			short i_f = 0;
			// Write sub block back to correct output location
			short x,y;
			x=y=0;
			int stride_div = stride==2?1:0;

			float div_sqrt_variance[STRIPES];
			float rolling_mean[STRIPES];
			float scales[STRIPES];
			float biases[STRIPES];
			float b_scales[STRIPES];

			for (char i = 0; i < STRIPES; i++)
			{
				float values[4];
				#pragma unroll
				for (int p = 0;p < 4; p++)
					values[p] = data[((o_f+i)<<2) + p];

				#pragma unroll
				for (int p = 0; p < (STRIPES-1);p++)
				{
					div_sqrt_variance[p] = div_sqrt_variance[p+1];
					rolling_mean[p] = rolling_mean[p+1];
					scales[p] = scales[p+1];
					biases[p] = biases[p+1];
				}
				div_sqrt_variance[(STRIPES-1)] = values[0];
				rolling_mean[(STRIPES-1)] = values[1];
				scales[(STRIPES-1)] = values[2];
				biases[(STRIPES-1)] = values[3];
				b_scales[i] = binary_scale[o_f + i];
			}


			while (y < sub_block_height)
			{
				// Index in output
				// First feature output
				// Read input from channels
				int input[STRIPES];

#pragma unroll STRIPES
				for (int p = 0; p < STRIPES; p++)
					input[p] = read_channel_intel(input_channels[p]);

				total_reads++;
				unsigned int out_index = o_f * (out_size*out_size) >> STRIPES_DIV;
				unsigned int in_index = (x + (y*sub_block_width));

				// Check y in bounds
				bool inbounds = ((y+sub_block_y_index)>>stride_div) < out_size ? true : false;
				out_index += (((x+sub_block_x_index)>>stride_div) + ((y+sub_block_y_index)>>stride_div) * out_size);

				float leaky_activation[STRIPES];
				
				#pragma unroll STRIPES
				for (int q = 0; q < STRIPES; q++)
				{
					float norm;
					float bias;
					float scale;
					float val = (float)(input[q]) *b_scales[q]; // / BINARY_FLOAT_SCALE; part of binary scale

					if (batch_normalised == 1)
					{
						norm =  (val - rolling_mean[/*o_f + */q])*div_sqrt_variance[/*o_f + */q];
						scale = scales[/*o_f + */q];
					}
					else
					{
						norm = val;
						scale =1.0f;
					}
					bias = norm*scale + biases[/*o_f + */q];
					float scaler = activation==FPGA_LEAKY?0.1f:1.0f;
					leaky_activation[q] = bias < 0.0?scaler*bias:bias;

				}
				if (inbounds)
				#pragma unroll STRIPES
					for (int q = 0; q < STRIPES; q++)
					output[(out_index<<STRIPES_DIV)+q] =leaky_activation[q];// (float)out_cache[in_index][q] *binary_scale[o_f + q]; // / BINARY_FLOAT_SCALE; part of binary scale
#ifndef ALTERA_CL
				out_count++;
#endif
				x = x != (sub_block_width-stride)?x+stride:0;
				y = x == 0 ? y+stride : y;
			}
		}

		sub_block_x = sub_block_x != (no_sub_blocks_x - 1) ? sub_block_x + 1 : 0;
		sub_block_y = sub_block_x == 0 ? sub_block_y+1 : sub_block_y;
	}
#ifndef ALTERA_CL
	if (input_channels[0].count != 0)
	{
		printf("channel should be empty\n");
		exit(1);
	}
#endif
}

