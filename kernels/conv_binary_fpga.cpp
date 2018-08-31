/*
 * basic_convolution_striped.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 *
 */


#include "conv_binary_fpga.h"

#include "fpga_channels.h"
#include "bnn_libraries.h"
#ifndef ALTERA_CL
#include <stdlib.h>
#endif

#ifdef ALTERA_CL
//#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
#endif

#pragma OPENCL EXTENSION cl_intel_channels : enable




/*
 * Coefficient kernel
 */

#ifdef ALTERA_CL
channel vec_uint coeff_channel_rdmem_binary2 __attribute__((depth(511)));

#else
CHAN(vec_uint,2) coeff_channel_rdmem_binary2;

#endif


#ifdef ALTERA_CL
channel vec_int input_channels __attribute__((depth(511)));

#else
CHAN(vec_int,8) input_channels;

#endif

#ifdef ALTERA_CL
__kernel
#else
extern "C"
#endif
void conv_coeffs_binary_subblock_fpga(
#ifdef ALTERA_CL
					   __global vec_uint *restrict coeffs,
#else
					   vec_uint *coeffs,
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
	short i,j,o_f,i_f,h,w,p,q;
	unsigned short y = 0;
	//for (int y = 0; y < y_div; y++)
	o_f = 0;
	unsigned int c;
	unsigned int c_cached;
	c = 0;
	i = 0;
	unsigned short ncoeffs = block_size;//+STRIPES;//(in_f*kernel_size*kernel_size+STRIPES);
	vec_uint Buffer[512];
	while (y < y_div)
	{
		unsigned short buffer_count = 0;
		// Important for overall performance do not remove.
		// Buffer locally so burst of data is possible.
		while ((y < y_div) && (buffer_count < 512))
		{
			vec_uint v;
			v = coeffs[c];
			Buffer[buffer_count] = v;
			i = (i < (ncoeffs-STRIPES)) ? i+STRIPES:0;
			o_f = (i == 0)? ((o_f != (out_f-STRIPES))?o_f+STRIPES:0):o_f;
			y = ((o_f == 0) && (i == 0))?y + 1: y;
			c = ((o_f == 0) && (i == 0))? 0 : c+1;
			buffer_count++;
		}
		for (short b = 0; b < buffer_count; b++)
			write_channel_intel(coeff_channel_rdmem_binary2,Buffer[b]);

	}
}




#ifndef ALTERA_CL
int out_count = 0;

float max_val = 0.0f;
float min_val = 0.0f;
#endif

#ifdef CREATE_TEST_VECTORS
// Functions used to create test vectors during HDL IP development

#include <stdio.h>
FILE *data_vecs = NULL;
FILE *weight_vecs = NULL;
FILE *result_vecs = NULL;
int total_vectors = 0;
#define MAX_VECTORS (1000*3)
unsigned int convolution_layer = 0;

void SetupTestVectorFiles()
{
	char filename1[100];
	char filename2[100];
	char filename3[100];
	sprintf(filename1,"D:/AHDL_Projects/BNN_IP/BNN_IP/src/TestBench/data_vectors_%d.txt",convolution_layer);
	sprintf(filename2,"D:/AHDL_Projects/BNN_IP/BNN_IP/src/TestBench/weight_vectors_%d.txt",convolution_layer);
	sprintf(filename3,"D:/AHDL_Projects/BNN_IP/BNN_IP/src/TestBench/result_vectors_%d.txt",convolution_layer);
	data_vecs = fopen(filename1,"w");
	weight_vecs = fopen(filename2,"w");
	result_vecs = fopen(filename3,"w");
	convolution_layer++;
	total_vectors = 0;
}

void CloseTestVectorFiles()
{
	fclose(data_vecs);
	fclose(weight_vecs);
	fclose(result_vecs);
}

void StoreVector(FILE *file,unsigned int bits,unsigned char *c,unsigned int bit_size)
{
	if (total_vectors < MAX_VECTORS)
	{
		for (int i = (bits-bit_size); i >= 0; i-=bit_size)
		{
			if (bit_size == 8)
			{
				unsigned char t = c[i/bit_size];
				fprintf(file,"%x",(t&0xf0)>>4);
				fprintf(file,"%x",(t&0xf));
			}
			if (bit_size == 16)
			{
				unsigned short t = ((unsigned short*)c)[i/bit_size];
				fprintf(file,"%x",(t&0xf000)>>12);
				fprintf(file,"%x",(t&0xf00)>>8);
				fprintf(file,"%x",(t&0xf0)>>4);
				fprintf(file,"%x",(t&0xf));
			}
			if (bit_size == 32)
			{
				unsigned int t = ((unsigned int*)c)[i/bit_size];
				fprintf(file,"%x",(t&0xf0000000)>>28);
				fprintf(file,"%x",(t&0xf000000)>>24);
				fprintf(file,"%x",(t&0xf00000)>>20);
				fprintf(file,"%x",(t&0xf0000)>>16);
				fprintf(file,"%x",(t&0xf000)>>12);
				fprintf(file,"%x",(t&0xf00)>>8);
				fprintf(file,"%x",(t&0xf0)>>4);
				fprintf(file,"%x",(t&0xf));
			}
		}
		fprintf(file,"\n");
		total_vectors++;
	}
}

#endif




inline vec_int AccumulateStripes(vec_uint weights,vec_short input,vec_int out,
					          unsigned short i_f,unsigned short l_c)
{
#ifdef CREATE_TEST_VECTORS
	StoreVector(data_vecs,(32*16),(unsigned char *)&input,16);
	StoreVector(weight_vecs,(32*32),(unsigned char *)&weights,32);
#endif
	vec_int sum;
#ifdef USE_LIBS
#ifdef STRIPES_32
	union
	{
		uint16 vec;
		vec_short input;
	}data_in;
	data_in.input = input;
	union
	{
		ulong16 vec;
		vec_uint input;
	}weights_in;
	weights_in.input = weights;
	union
	{
		ulong16 vec;
		vec_int input;
	}out_in;	
	union
	{
		ulong16 vec;
		vec_int input;
	}dataout_in;
	out_in.vec = BCNNx16BITx32(weights_in.vec,data_in.vec);
#pragma unroll STRIPES
	for (int q = 0; q < STRIPES; q++)
		dataout_in.input.s[q] = out.s[q] + out_in.input.s[q];
	return dataout_in.input;
#endif
#else
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
			res += (add_sub2&0x1)==1?value:-value;
		}
		sum.s[q] = out.s[q] + res;
	}
#ifdef CREATE_TEST_VECTORS
	StoreVector(result_vecs,(32*32),(unsigned char *)&sum,32);
#endif
	return sum;
#endif

}



#ifdef ALTERA_CL
__kernel
#else
//extern "C"
int clocks=0;

#endif
void conv_binary_subblock_fpga_v4(
#ifdef ALTERA_CL
					   __global vec_float  *restrict input,
					   __global unsigned int *restrict coeffs,
					   __global float *restrict output,
#else
					   vec_float *input, unsigned int *coeffs, float *output,
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
					   unsigned int total_block_count,
					   unsigned char power // power to raise input by
					  ) // Reduces the amount of output cahce memory required
{
#ifdef CREATE_TEST_VECTORS
	SetupTestVectorFiles();
	clocks = 0;
#endif
	// coefficients are double buffered and read from parallel kernel
	// this allows time when memory is quiet to cache coefficients ahead of time
	// hopefully reducing any bottlenecks

	// No longer splitting in Y as this is probablematic for large images with odd
	// striding

	// Now splitting image into blocks with halo big enough for stride and filter size

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
	unsigned char kernel_size_sq = (unsigned char)kernel_size*(unsigned char)kernel_size;

#ifdef ALTERA_CL
	vec_short img_cache0[MAX_INPUT_IMAGE_BATCH];
	vec_short img_cache1[MAX_INPUT_IMAGE_BATCH];
	vec_int out_cache0[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
	vec_int out_cache1[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
	vec_int out_cache2[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
	vec_int out_cache3[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
#else
	auto img_cache0 = new 	vec_short[MAX_INPUT_IMAGE_BATCH];
	auto img_cache1 = new 	vec_short[MAX_INPUT_IMAGE_BATCH];
	auto out_cache0 = new 	vec_int[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
	auto out_cache1 = new 	vec_int[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
	auto out_cache2 = new 	vec_int[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
	auto out_cache3 = new 	vec_int[64+ MAX_OUTPUT_IMAGE_BATCH>>PAR_PIX_SHIFT];
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

		short pad = ((kernel_size-1)>>1);
		short pad2 = pad<<1;
		short copy_width = sub_block_width + pad2;
		short copy_height = sub_block_height + pad2;
		unsigned int c = 0;

		short i_f;
		short x,y;
		x = y =  -pad;
		i_f = 0;

		while (i_f < in_f)
		{
			unsigned int feature_offset = (i_f * size * size)>>STRIPES_DIV;
			int index_x = x + sub_block_x_index;
			int index_y = y + sub_block_y_index;
			bool out_of_bounds = false;
			if ((index_x < 0) || (index_y < 0) || (index_x >= size) || (index_y >= size))
				out_of_bounds = true;
			int index = feature_offset + ((index_x) + (index_y * size));

			vec_float row;
			vec_short row_conv;
			if (!out_of_bounds)
				row = input[index];

			#pragma unroll
			for (int p = 0; p < STRIPES; p++)
				row_conv.s[p] = out_of_bounds ? 0 : FloatToShort(row.s[p],power);
			img_cache0[c] = row_conv;
			img_cache1[c] = row_conv;
			c++;
			x = (x != (sub_block_width+pad-1))?x+1:-pad;
			y = (x == -pad) ? ((y != (sub_block_height+pad-1))?y+1:-pad):y;
			i_f = ((x==-pad) && (y == -pad))?i_f+STRIPES:i_f;
#ifndef ALTERA_CL
			clocks+=1; // Limited by 1 DDR bank
#endif
		}
#ifdef ALTERA_CL
		mem_fence(CLK_LOCAL_MEM_FENCE); // make sure execution waits before running next loop.
#endif
	
		// Input is now cached. No loop over all possible outputs for this sub block
		// Variable to count in coefficients
		unsigned int coefficent_load_count = 0;
		// Variables for locating position in input filter
		short x2;
		x = 0; y = 0;
		unsigned char stride_shift = ((stride==2)?1:0);

		unsigned short block_size = (sub_block_width>>stride_shift) * ( sub_block_height>>stride_shift);
		// Make sure block size is rounded up to par pixels
		unsigned short block_size_rounded = (block_size&PAR_PIX_MASK)?block_size + (PAR_PIX - block_size&PAR_PIX_MASK):block_size;

		unsigned short block_size_shifted = block_size >> STRIPES_DIV;
		unsigned short input_block_size = sub_block_width*sub_block_height;
		unsigned short block_chunk = (copy_width * copy_height);
		short o_f = 0;
		char w,h;
		w=-1;
		h= 0;
		i_f = 0;
		c = 0;
		//printf("start block\n");
		int feature_offset = -block_chunk;
		int block_count = -1;
		unsigned short filter_blocks = in_f>>STRIPES_DIV;
		unsigned short out_filter_blocks = out_f>>STRIPES_DIV;
		filter_blocks = filter_blocks == 0 ? 1 : filter_blocks;

		short w_inc_counter = -1;
		short w_zero_point = 1;

		short h_inc_counter = -1;
		short h_zero_point = kernel_size;

		short i_f_inc_counter = -1;
		unsigned short i_f_zero_point = kernel_size_sq;

		short o_f_inc_counter = -1;
		unsigned short o_f_zero_point = kernel_size_sq*filter_blocks;
		unsigned short sub_block_width_shifted = sub_block_width>>stride_shift;
		char ww,hh;
		ww=-1;
		hh=-1;
		short i_ff=-STRIPES;
		short o_ff=-block_size_rounded;
		vec_int zero;
		#pragma unroll
		for (int p = 0; p < STRIPES; p++)
			zero.s[p] = 0;

		//unsigned int total_block_count = filter_blocks*kernel_size_sq*(out_f>>STRIPES_DIV)*block_size;
		//for (int i = 0; i <total_block_count ;i++)
		//unsigned int i = 0;
		int cc = -PAR_PIX;
		//short cc_test = -PAR_PIX;
#ifdef ALTERA_CL
#pragma ivdep array(out_cache0)
#pragma ivdep array(out_cache1)
#pragma ivdep array(out_cache2)
#pragma ivdep array(out_cache3)
#endif
		for (unsigned int i = 0; i < total_block_count; i++)
		{
			// Load coefficients in parallel using double buffer
			cc = cc < (block_size-PAR_PIX)?cc+PAR_PIX:0;
			block_count = (c == 0) ? block_count+1:block_count;

			w_inc_counter = (cc == 0) ? (w_inc_counter != (w_zero_point-1) ? w_inc_counter + 1 : 0): w_inc_counter;
			ww = (cc == 0) && (w_inc_counter == 0) ? ((ww != (kernel_size -1))? ww + 1 : 0) :  ww;

			h_inc_counter = (cc == 0) ? (h_inc_counter != (h_zero_point-1) ? h_inc_counter + 1 : 0): h_inc_counter;
			hh = (cc == 0) && (h_inc_counter == 0) ? ((hh != (kernel_size -1))? hh + 1 : 0) :  hh;

			i_f_inc_counter = (cc == 0) ? (i_f_inc_counter != (i_f_zero_point-1) ? i_f_inc_counter + 1 : 0): i_f_inc_counter;

			feature_offset =  (cc == 0) && (i_f_inc_counter == 0) ? ((i_ff != (in_f - STRIPES))? feature_offset + block_chunk : 0) :  feature_offset;

			i_ff = (cc == 0) && (i_f_inc_counter == 0) ? ((i_ff != (in_f - STRIPES))? i_ff + STRIPES : 0) :  i_ff;

			o_f_inc_counter = (cc == 0) ? (o_f_inc_counter != (o_f_zero_point-1) ? o_f_inc_counter + 1 : 0): o_f_inc_counter;
			o_ff = (cc == 0) && (o_f_inc_counter == 0) ? ((o_ff != (out_filter_blocks -1))? o_ff + block_size_rounded : 0) :  o_ff;

			if (cc == 0)
			{
					kernel_mask_A  = read_channel_intel(coeff_channel_rdmem_binary2);
			}
			// Caculate several pixels in parallel.
			unsigned short out_index = o_ff + cc>>PAR_PIX_SHIFT;

#pragma unroll PAR_PIX
			for (int q = 0; q < PAR_PIX; q++)
			{
				// Logic expensive but allows easily indexing for multiple pixel implementation!
#ifdef USE_LIBS
				// Efficient version of modulus divide using minimum number of bits.
				uchar2 indexes = mod_div_16bit_to_8bit(cc+q,sub_block_width_shifted);
				unsigned char x = (indexes.s0<<stride_shift);
				unsigned char y = (indexes.s1<<stride_shift);
#else
				unsigned char x = (((cc+q)%((unsigned short)sub_block_width_shifted))<<stride_shift);
				unsigned char y = (((cc+q)/((unsigned short)sub_block_width_shifted))<<stride_shift);
#endif

				unsigned short xx = x + ww;
				unsigned short yy = y + hh;

				unsigned short index = (xx + (yy*copy_width)) + feature_offset;
				//unsigned short out_index = o_ff + ((x>>stride_shift) + ((y>>stride_shift)*(sub_block_width_shifted)));

				vec_short input_vals;

				if ((q == 0) || (q == 1))
				{
//#pragma unroll STRIPES
//					for (int p = 0; p < STRIPES; p++) input_vals.s[p] = ((i_ff + p ) < l_c)?img_cache0[index].s[p]:0;
					#pragma unroll STRIPES
						for (int p = 0; p < STRIPES; p++) input_vals.s[p] = img_cache0[index].s[p];
				}
				if ((q == 2) || (q == 3))
				{
//#pragma unroll STRIPES
//					for (int p = 0; p < STRIPES; p++) input_vals.s[p] = ((i_ff + p ) < l_c)?img_cache1[index].s[p]:0;
#pragma unroll STRIPES
					for (int p = 0; p < STRIPES; p++) input_vals.s[p] = img_cache1[index].s[p];
				}

				vec_int out,in;
				vec_int out2,in2;
				switch (q)
				{
				case 0 : in = out_cache0[out_index]; break;
				case 1 : in = out_cache1[out_index]; break;
				case 2 : in = out_cache2[out_index]; break;
				case 3 : in = out_cache3[out_index]; break;
				default : break;
				}
				if ((hh<=0) && (ww <=0) && (i_ff == 0)) in = zero;

				out = AccumulateStripes(kernel_mask_A,input_vals,in,i_ff,l_c);
				if ((cc+q) < block_size)
				{
					switch(q)
					{
					case 0:	out_cache0[out_index] = out; break;
					case 1:	out_cache1[out_index] = out; break;
					case 2:	out_cache2[out_index] = out; break;
					case 3:	out_cache3[out_index] = out; break;
					default: break;
					}
				}
			}

#ifndef ALTERA_CL
			if ((cc&0xff) == 0)
			{
				//printf("in_f = %d, out_f  = %d : feature_offset = %d, c = %d,w = %d,h = %d, i_f = %d, o_f = %d\n",in_f,out_f,feature_offset,c,w,h,i_f,o_f);
				//printf("ww = %d, hh = %d\n",ww,hh);
			}
			clocks++;
#endif
			// Update indexes
			// Serveral counters created to reduce combinatorial depth and improved fmax.

			//i = c == 0 ? block_size+i : i;


			//w_inc_counter = (cc >= (block_size-PAR_PIX)) ? (w_inc_counter != (w_zero_point-1) ? w_inc_counter + 1 : 0): w_inc_counter;
			//ww = (cc >= (block_size-PAR_PIX)) && (w_inc_counter == 0) ? ((ww != (kernel_size -1))? ww + 1 : 0) :  ww;

			//h_inc_counter = (cc >= (block_size-PAR_PIX)) ? (h_inc_counter != (h_zero_point-1) ? h_inc_counter + 1 : 0): h_inc_counter;
			//hh = (cc >= (block_size-PAR_PIX)) && (h_inc_counter == 0) ? ((hh != (kernel_size -1))? hh + 1 : 0) :  hh;

			//i_f_inc_counter = (cc >= (block_size-PAR_PIX)) ? (i_f_inc_counter != (i_f_zero_point-1) ? i_f_inc_counter + 1 : 0): i_f_inc_counter;
			//feature_offset =  (cc >= (block_size-PAR_PIX)) && (i_f_inc_counter == 0) ? ((i_ff != (in_f - STRIPES))? feature_offset + block_chunk : 0) :  feature_offset;
			//i_ff = (cc >= (block_size-PAR_PIX)) && (i_f_inc_counter == 0) ? ((i_ff != (in_f - STRIPES))? i_ff + STRIPES : 0) :  i_ff;

			//o_f_inc_counter = (cc >= (block_size-PAR_PIX)) ? (o_f_inc_counter != (o_f_zero_point-1) ? o_f_inc_counter + 1 : 0): o_f_inc_counter;
			//o_ff = (cc >= (block_size-PAR_PIX)) && (o_f_inc_counter == 0) ? ((o_ff != (out_filter_blocks -1))? o_ff + block_size_rounded : 0) :  o_ff;

		}
		// Send data to channels when ready
		unsigned short sub_block = 0;
		unsigned int sub_block_count = 0;
		//for (int i = 0; i < block_size*out_filter_blocks;i++)
		while (sub_block_count < out_filter_blocks)
		{
			vec_int out;
			unsigned int index = sub_block + sub_block_count*block_size_rounded;
			switch (index&PAR_PIX_MASK)
			{
			case 0 : out = out_cache0[index>>PAR_PIX_SHIFT];break;
			case 1 : out = out_cache1[index>>PAR_PIX_SHIFT];break;
			case 2 : out = out_cache2[index>>PAR_PIX_SHIFT];break;
			case 3 : out = out_cache3[index>>PAR_PIX_SHIFT];break;
			default : break;
			}

			write_channel_intel(input_channels,out);
#ifndef ALTERA_CL
			clocks++;
#endif
			sub_block = sub_block != (block_size-1)?sub_block + 1:0;
			sub_block_count = sub_block == 0? sub_block_count +1 : sub_block_count;
		}

		sub_block_x = sub_block_x != (no_sub_blocks_x - 1) ? sub_block_x + 1 : 0;
		sub_block_y = sub_block_x == 0 ? sub_block_y+1 : sub_block_y;
	}
#ifndef ALTERA_CL
		delete out_cache0;
		delete out_cache1;
		delete out_cache2;
		delete out_cache3;
		delete img_cache0;
		delete img_cache1;
#endif
}



// Output kernel
#ifdef ALTERA_CL
__kernel
#else
//extern "C"
#endif
void conv_activations_v4(
#ifdef ALTERA_CL
					   __global vec_float *restrict output,
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
					  __global float *restrict data,
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


	float div_sqrt_variance[1024];
	float rolling_mean[1024];
	float scales[1024];
	float biases[1024];
	float b_scales[1024];

	for (unsigned short i = 0; i < out_f; i++)
	{
		float values[4];
		#pragma unroll
		for (int p = 0;p < 4; p++)
			values[p] = data[(i<<2) + p];

		div_sqrt_variance[i] = values[0];
		rolling_mean[i] = values[1];
		scales[i] = values[2];
		biases[i] = values[3];
		b_scales[i] = binary_scale[i];

	}
#ifdef ALTERA_CL
	mem_fence(CLK_LOCAL_MEM_FENCE);
#endif

	pad = ((kernel_size-1)>>1);
	int pad2 = pad<<1;
	int copy_width = sub_block_width + pad2;
	int copy_height = sub_block_height + pad2;
	int stride_div = stride==2?1:0;
	unsigned short block_size = (unsigned short)out_size*(unsigned short)out_size;

	while (sub_block_y != no_sub_blocks_y)
	{
		// Temporary input buffer. Converted to short of binary addition calculation
		// Each block needs contain all input features
		//int out_cache[104*104][STRIPES];

		// Load in required data
		unsigned short sub_block_x_index = (sub_block_x * sub_block_width);
		unsigned short sub_block_y_index = (sub_block_y * sub_block_height);


		// Input is now cached. No loop over all possible outputs for this sub block
		// Variable to count in coefficients

		unsigned short stride_div = stride==2?1:0;
		unsigned short o_f = 0;

		//for (short o_f = 0; o_f < out_f; o_f +=STRIPES)
		unsigned short x,y;
		unsigned short yt = 0;
		x=y=0;
		unsigned int out_index_offset = 0;
		unsigned int loop_size = (sub_block_width>>stride_div)* (sub_block_height>>stride_div)*(out_f>>STRIPES_DIV);


		unsigned short y_inc_counter = 0;
		unsigned short y_zero_point = (sub_block_width)>>stride_div;

		unsigned short o_f_inc_counter = 0;
		unsigned short o_f_zero_point = ((sub_block_width)>>stride_div)*((sub_block_height)>>stride_div);

		//while (o_f < out_f)
		for (unsigned int l = 0; l < loop_size; l++)
		{
			// Write sub block back to correct output location

			//while (y < sub_block_height)
			//{
				vec_float __attribute__((register)) leaky_activation;
				// Index in output
				// First feature output
				// Read input from channels
				vec_int input;

				input = read_channel_intel(input_channels);

				// Check y in bounds
				bool inbounds = ((y+sub_block_y_index)>>stride_div) < out_size ? true : false;
				///unsigned int out_index_offset =  o_f * (out_size*out_size) >> STRIPES_DIV;
				unsigned int out_index_pixel = (((x+sub_block_x_index)>>stride_div) +
						(unsigned int)((unsigned short)((y+sub_block_y_index)>>stride_div) * (unsigned short)out_size));
				unsigned int out_index = out_index_offset + out_index_pixel;

				
				#pragma unroll STRIPES
				for (int q = 0; q < STRIPES; q++)
				{
					float norm;
					float bias;
					float scale;
					float val = (float)(input.s[q]) *b_scales[o_f + q]; // / BINARY_FLOAT_SCALE; part of binary scale

					if (batch_normalised == 1)
					{
						norm =  (val - rolling_mean[o_f + q])*div_sqrt_variance[o_f + q];
						scale = scales[o_f + q];
					}
					else
					{
						norm = val;
						scale =1.0f;
					}
					bias = norm*scale + biases[o_f + q];
					float scaler = activation==FPGA_LEAKY?0.1f:1.0f;
					leaky_activation.s[q] = bias < 0.0f?scaler*bias:bias;

				}
				if (inbounds)
				{
#ifdef ALTERA_CL
					output[out_index] = leaky_activation;// (float)out_cache[in_index][q] *binary_scale[o_f + q]; // / BINARY_FLOAT_SCALE; part of binary scale
#else
					#pragma unroll STRIPES
					for (int q = 0; q < STRIPES; q++)
						output[(out_index<<STRIPES_DIV)+q] =leaky_activation.s[q];// (float)out_cache[in_index][q] *binary_scale[o_f + q]; // / BINARY_FLOAT_SCALE; part of binary scale
#endif
				}

#ifndef ALTERA_CL
				out_count++;
#endif
				x = x != (sub_block_width-stride)?x+stride:0;
				//y = x == 0 ? ((y == (sub_block_height-stride))?0:y+stride) : y;
				y_inc_counter = (y_inc_counter != (y_zero_point-1)) ? y_inc_counter + 1 : 0;
				y = (y_inc_counter == 0) ? ((y == (sub_block_height-stride))?0:y+stride):y;

				//o_f = (x==0) && (y==0) ? o_f += STRIPES:o_f;
				o_f_inc_counter = (o_f_inc_counter != (o_f_zero_point-1)) ? o_f_inc_counter + 1 : 0;
				o_f = (o_f_inc_counter == 0) ? o_f+STRIPES:o_f;

				out_index_offset = (o_f_inc_counter == 0) ? (out_index_offset + block_size) : out_index_offset;
			//}
		}




		sub_block_x = sub_block_x != (no_sub_blocks_x - 1) ? sub_block_x + 1 : 0;
		sub_block_y = sub_block_x == 0 ? sub_block_y+1 : sub_block_y;
	}
#ifndef ALTERA_CL
	#ifdef CREATE_TEST_VECTORS
		 CloseTestVectorFiles();
	#endif
	printf("clocks = %d : time = %f\n",clocks,clocks/155000000.0f);
	if (input_channels.count != 0)
	{
		printf("channel should be empty\n");
		exit(1);
	}
#endif
}

