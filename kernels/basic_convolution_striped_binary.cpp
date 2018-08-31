/*
 * basic_convolution_striped.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 *
 */


#include "basic_convolution_striped_binary.h"
#include "fpga_channels.h"
//#define USE_HDL_LIBRARIES
#ifdef USE_HDL_LIBRARIES
#ifdef ALTERA_CL
#include "bnn_lib.h"
#endif
#endif

#pragma OPENCL EXTENSION cl_intel_channels : enable

#define FIXED_POINT
// 8BIT
#define FIXED_POINT_MULTIPLIER 9
//#define FIXED_POINT_MULTIPLIER 8
#define FIXED_POINT_DIVIDER 4

#ifdef FIXED_POINT
#define f_type int
#define inter_type int
#else
#define f_type float
#define inter_type float
#endif

/*
 * Coefficient kernel
 */

#ifdef ALTERA_CL
channel unsigned int coeff_channel_rdmem_binary_bits __attribute__((depth(1024)));
#else
CHAN(unsigned int,1024) coeff_channel_rdmem_binary_bits;

#endif



#ifdef ALTERA_CL
__kernel
#else
extern "C"
#endif
void basic_convolution_striped_load_coeffs_kernel_binary_bits(
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
								last = (y == (y_div-1)) &&
										(out_batch == (batches_of_49-1)) &&
												(i_f == (in_f-STRIPES)) &&
												(o_f == (out_f-STRIPES))&&
												(h ==  (kernel_size-pad-1))? 1: 0;
								for ( w =-pad; w < (kernel_size-pad+last); w++)
								{
									for ( p = 0; p < STRIPES; p++)
									{
										unsigned int bits = coeffs[c++];
										write_channel_intel(coeff_channel_rdmem_binary_bits,bits);
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

#ifndef ALTERA_CL
inline int nalla_ldexp_cast(float input,char exp)
{
	union
	{
		float f;
		unsigned int i;
	}t;
	t.f = input;
	if (input != 0)
	{
		unsigned int frac = 0x800000|(t.i&0x7fffff);
		unsigned int exp2 = 126-((0x7f800000&t.i) >> 23);
		unsigned int shift = ((24+exp2)-exp);
		unsigned int frac2 = frac >> (shift > 31 ? 31 : shift);
		int sign = (t.i&0x80000000)?-frac2:frac2;
		int normal = ldexp(input,exp);
		if (sign != normal)
			printf("error\n");
		return sign;
	}
	else
		return 0;
	//return (int)ldexp(input,exp);
}

float nalla_ldexp(float input,char exp)
{

	return ldexp(input,exp);
}
#else
char nalla_ldexp_cast(float input,char exp)
{
	// HDL IP
	char res;
	res = FloatToCharExp(input,exp);
	return res;
}

inline float nalla_ldexp(float input,char exp)
{

	return ldexp(input,exp);
}
#endif


int GetBit(unsigned int val,int id)
{
	return ((val>>id)&0x1);
}

/*
 * Use HDL for bit counting.
 * Striped 32
 */

#ifndef ALTERA_CL

#endif
#ifdef CREATE_TEST_VECTORS
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
			unsigned char t = c[i/bit_size];
			if (bit_size == 8)
			{
				fprintf(file,"%x",(t&0xf0)>>4);
				fprintf(file,"%x",(t&0xf));
			}
			if (bit_size == 16)
			{
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

#ifdef USE_HDL_LIBRARIES
inline void CountBitsHDL32(bool toggle,
					unsigned int *bits_A,
					unsigned int *bits_B,
					float input[MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES],
					unsigned int index,
					char exponent_shifts,
					bool accumulate_output,
					f_type out_cache[MAX_BATCH_SIZE][STRIPES],
					unsigned int out_index
					)
{
	// Create input vectors for HDL
	union
	{
#ifdef ALTERA_CL
		ulong16 coeffs_hdl;
#endif
		unsigned int bits[32];
	} coeffs_union;
#pragma unroll
	for (int p = 0; p < STRIPES; p++)
		coeffs_union.bits[p] = toggle?bits_A[p]:bits_B[p];

	union
	{
#ifdef ALTERA_CL
		ushort16 data_hdl;
#endif
		char data[32];
	} data_union;

	union
	{
#ifdef ALTERA_CL
		uint16 result_hdl;
#endif
		char result[32];
	} result_union;


#pragma unroll
	for (int p = 0; p < STRIPES; p++)
	{
		data_union.data[p] = nalla_ldexp_cast(input[index][p],exponent_shifts);
	}
#ifdef CREATE_TEST_VECTORS
	StoreVector(data_vecs,8*32,(unsigned char*)data_union.data);
	StoreVector(weight_vecs,32*32,(unsigned char*)coeffs_union.bits);
#endif

#ifdef ALTERA_CL
	result_union.result_hdl = BCNNx8BITx32(coeffs_union.coeffs_hdl,data_union.data_hdl);
	#pragma unroll
	for (int p = 0; p < STRIPES; p++)
		if (accumulate_output)
			out_cache[out_index][p] += result_union.result[p];
		else
			out_cache[out_index][p] = result_union.result[p];
#else
	// Soft simulation to test
	// Un-comment to test

#ifdef CREATE_TEST_VECTORS
	short result_data[32];
#endif
	#pragma unroll
	for (int q = 0; q < STRIPES; q++)
	{
		inter_type res = 0;
		#pragma unroll
		for (int p = 0; p < STRIPES; p++)
		{
			inter_type val = data_union.data[p];
			inter_type value = val;
				int add_sub = GetBit(coeffs_union.bits[q],p);
			res += (add_sub&0x1)==1?value:-value;
		}
#ifdef CREATE_TEST_VECTORS
		result_data[q] = res;
#endif
		if (!accumulate_output)//(h<=-pad) && (w ==-pad)))
			out_cache[out_index][q] = res;
		else
			out_cache[out_index][q] +=res;
	}
#ifdef CREATE_TEST_VECTORS
	StoreVector(result_vecs,16*32,(unsigned char*)result_data);
#endif

#endif


}
#endif


#ifdef ALTERA_CL
__kernel
#else
//extern "C"
#endif
void basic_convolution_striped_kernel_binary(
#ifdef ALTERA_CL
					   __global float *restrict input,
					   __global float *restrict scale, // binary output scaler
					   __global float *restrict output,
#else
					   float *input, float *scale, float *output,
#endif
					   int batch,
					   int groups,
					   int nweights,
					   int size,
					   int out_size,
					   int size_div_y_div,
					   int out_size_div_y_div,
					   int kernel_size,
					   int pad,
					   int in_f,
					   int true_in_f, // true size if scaled
					   int out_f,
					   int stride,
					   int batches_of_49,
					   int y_div) // Reduces the amount of output cahce memory required
{
	// coefficients are double buffered and read from parallel kernel
	// this allows time when memory is quiet to cache coefficients ahead of time
	// hopefully reduing any bottlenecks

	unsigned int kernel_mask_A_bits[STRIPES];
	unsigned int kernel_mask_B_bits[STRIPES];
#ifdef CREATE_TEST_VECTORS
	SetupTestVectorFiles();
#endif


#ifdef FIXED_POINT
#ifndef ALTERA_CL
	auto img_cache = new float[ MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES];
	auto out_cache = new f_type[ MAX_BATCH_SIZE][STRIPES];
#else
	float img_cache[MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES];
	f_type out_cache[MAX_BATCH_SIZE][STRIPES];
#endif
#else
#ifndef ALTERA_CL
	auto img_cache = new float[ MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES];
	auto out_cache = new float[ MAX_BATCH_SIZE][STRIPES];
#else
	float img_cache[MAX_INPUT_IMAGE_BATCH +MAX_PADDING_SPACE][STRIPES];
	float out_cache[MAX_BATCH_SIZE][STRIPES];
#endif
#endif

	short ksize =  (kernel_size*kernel_size);
	char first = 1;
	short sizex,sizey;
	short out_sizex,out_sizey;
	sizex = size;
	sizey = size_div_y_div;//(short)size/(short)y_div; // Move to host
	out_sizex = out_size;
	out_sizey = out_size_div_y_div;//(short)out_size/(short)y_div;
	short y_div_offset = 0;
	bool toggle = 0;
	unsigned int out_y_div_offset=0;
	char pad_compensate = pad;
	unsigned short offset_increment = ((sizex+(pad<<1))*(sizey+(pad_compensate<<1)));

#define MAX_KERNEL_SIZE (11*11)
	/*
	 * Create a lookup table for coefficient offsets
	 * Makes main loop much more efficient. Don't need to track w and h
	 *
	 * To avoid checking if out of bounds the input buffer is created with
	 * a halo of zeros of size pad.
	 *
	 * Vertical padding is already done for dividing input image in y axis.
	 */
	short index_offsets[MAX_KERNEL_SIZE];
	char c = 0;
	for (char h = 0; h <= (pad<<1); h++)
		for (char w = 0; w <= (pad<<1); w++)
		{
			index_offsets[c++] = w + (h*(size+(pad<<1)));
		}

    for (int yy_div = 0; yy_div < y_div; yy_div++)
	{
		y_div_offset = yy_div*sizey;
		unsigned int c,d;
		c=d=0;
		// Get max exponent as data loaded
		char max_exponents[STRIPES];// Do this on a feature by feature basis
		short exponent_shifts;// Do this on a feature by feature basis
		#pragma unroll
		for (int p = 0; p < STRIPES; p++)
			max_exponents[p] = -99;

		// Add vertical padding
		for (short i_f = 0; i_f  < in_f; i_f+=STRIPES)
		{
			short x,y;
			x = -pad; // Start from -pad for halo
			y = -pad_compensate;
			int input_index = (i_f*size*size)+  (yy_div*sizey*sizex<<STRIPES_DIV);
			int s = (i_f*size*size) + ((yy_div*sizey-pad_compensate)*(sizex<<STRIPES_DIV));
			while (y < (sizey+pad_compensate))
			{
				int input_index2 = s;
				// check for halo
				bool halo = ((x < 0) || (x >= size) || (y >= size) || (y < 0))?true:false;
				unsigned char exp[STRIPES];
				#pragma unroll
				for (int p = 0; p < STRIPES; p++)
				{
					union {
						float f;
						unsigned int i;
					}val;
					val.f = halo?0:(input_index2 < 0?0:input[input_index2+p]);
					char exp  = (((val.i>>23)&0xff)-127);
					if ((exp > max_exponents[p]) && (val.i != 0))
						max_exponents[p] = exp;

#ifdef FIXED_POINT
					img_cache[d][p] = val.f;//(inter_type)(nalla_ldexp(val,FIXED_POINT_MULTIPLIER));
					//img_cache[d][p] = (f_type)(val);
#else
					img_cache[d][p] = (f_type)(val.f);
#endif
				}
#pragma unroll

				if ((x >= 0) && (x < sizex))
					s+=STRIPES;
				x = x != (sizex+pad-1)? x+1:-pad;
				y = x == -pad ? y+1:y;
				d++;
			}
		}
		char max_exponent = -99;
#pragma unroll
		for (int e = 0; e < STRIPES;e++)
		{
			if (max_exponents[e] > max_exponent)
				max_exponent = max_exponents[e];
		}
		exponent_shifts = FIXED_POINT_MULTIPLIER  - max_exponent;
		// Coefficient count
		c = 0;
		unsigned int o=0;
		unsigned output_offset = 0;

		for (short o_f = 0; o_f < out_f; o_f +=STRIPES)
		{
			unsigned short coefficent_load_count = 0;
			unsigned short batch_pointer = 0;
			unsigned int out_batch_offset = 0;
			// Load as registers using shift
			float scalers[STRIPES];
			for (int q = 0; q < STRIPES; q++)
			{
#pragma unroll
				for (int p = 0; p < (STRIPES-1); p++)
					scalers[p] = scalers[p+1];
				scalers[(STRIPES-1)] = scale[q+o_f];
			}


			for (short out_batch = 0; out_batch < batches_of_49; out_batch++)
			{
				short x,y;
				short xx,yy;
				unsigned int in_offset = 0;//(j*in_f*sizex*sizey) + (i*groups*in_f*sizex*sizey);
				short out_batch_count = 0;
				short i_f = 0;
				xx = batch_pointer%(sizex);
				yy = (batch_pointer/(sizex))*stride;
				x = xx;
				y = yy;
				short index_offset_count = first?-1:0; // forces extra loop for kernel setup
				first = false;
				#pragma ivdep
				while (i_f < in_f)
				{
					/*
					 * Load coefficients whilst processing previous in double buffer fashion.
					 */
					if (coefficent_load_count < STRIPES)
					{
						unsigned int vec_bits = read_channel_intel(coeff_channel_rdmem_binary_bits);
						if (toggle)
						{
							#pragma unroll
							for (int p = 0; p < (STRIPES-1);p++)
							{
								kernel_mask_B_bits[p] = kernel_mask_B_bits[p+1];
							}
							kernel_mask_B_bits[(STRIPES-1)] = vec_bits;
						}
						else
						{
							#pragma unroll
							for (int p = 0; p < (STRIPES-1);p++)
							{
								kernel_mask_A_bits[p] = kernel_mask_A_bits[p+1];
							}
							kernel_mask_A_bits[(STRIPES-1)] = vec_bits;
						}
					}

					//bool out_of_bounds = ((h+y+y_div_offset) < 0) || ((w+x) < 0) || ((h+y+y_div_offset) >= size) || ((w+x) >= size);
					//short index = (w+x+pad) + ((h+(y+pad_compensate))*(sizex+(pad<<1))) + in_offset; // Compensate for extra padding held in memory
					short index = 0;
					if (index_offset_count >= 0)
					{
						//index = index_offsets[(h+pad)*((pad<<1)+1)+w+pad] + in_offset + x + y*(sizex+(pad<<1));
						index = index_offsets[index_offset_count] + in_offset + x + y*(sizex+(pad<<1));
					}

#ifdef USE_HDL_LIBRARIES
     CountBitsHDL32(toggle,
					kernel_mask_A_bits,
					kernel_mask_B_bits,
					img_cache,
					index,
					exponent_shifts,
					((i_f == 0) && (index_offset_count==0))?false:true,
					out_cache,
					out_batch_count
					);
#else

#pragma unroll
					for (int q = 0; q < STRIPES; q++)
					{
						inter_type res = 0;
						unsigned int bits = (toggle?kernel_mask_A_bits[q]:kernel_mask_B_bits[q]);

						#pragma unroll
						for (int p = 0; p < STRIPES; p++)
						{
#ifndef ALTERA_CL
							inter_type val = (inter_type)(nalla_ldexp_cast(img_cache[index][p],exponent_shifts));
							//inter_type value = (inter_type)(nalla_ldexp(val,FIXED_POINT_MULTIPLIER));
							//inter_type value = (inter_type)(index >= 0?img_cache[index][p]:0);
							inter_type value = val;
#else
#ifdef NO_EMU
							inter_type value = (inter_type)img_cache[index][p];
#else
							inter_type value =(inter_type)(index >= 0?img_cache[index][p]:0);
#endif
#endif

							int add_sub = GetBit(bits,p);
							res += (add_sub&0x1)==1?value:-value;
						}
						if ((i_f == 0) && (index_offset_count==0))//(h<=-pad) && (w ==-pad)))
							out_cache[out_batch_count][q] = res;
						else
							out_cache[out_batch_count][q] +=res;
					}
#endif

					// Calculate all indexes
					out_batch_count = out_batch_count != (MAX_BATCH_SIZE-1)?out_batch_count+1:0;
					if ((out_batch_count==0) && (index_offset_count == (ksize-1)))//(h == (kernel_size-pad-1)) && (w == (kernel_size-pad-1)))
					{
						in_offset+= offset_increment;// (sizex*(sizey+(pad_compensate<<1)));
						i_f+= STRIPES;
					}
					if (out_batch_count == 0)
						index_offset_count = index_offset_count != (ksize-1)? index_offset_count +1 :0;

					x = (out_batch_count==0) ? xx : (x != (sizex-stride)? x+stride:0);
					y =  (out_batch_count==0) ? yy : ((x == 0)? (y != ((sizey+pad_compensate+pad_compensate)-stride)?y+stride:0):y);

					coefficent_load_count = (out_batch_count==0)? 0 : coefficent_load_count+1;

					toggle =(out_batch_count==0)?(toggle?0:1):toggle;
				}
				unsigned short cache=0;
				//unsigned int oo = (o_f *out_size*out_size) + (STRIPES*out_batch*MAX_BATCH_SIZE) + (STRIPES*yy_div*out_sizey*out_sizex);
				unsigned int oo = output_offset + out_batch_offset + out_y_div_offset;
				for ( cache = 0; cache < MAX_BATCH_SIZE; cache++)
				{
					#pragma unroll
					for (int q = 0; q < STRIPES; q++)
					{
						union
						{
							unsigned int i;
							float f;
						}out_res;
#ifdef FIXED_POINT
						out_res.f = (float)out_cache[cache][q]; // Keeps accuracy better!

						out_res.f = nalla_ldexp(out_res.f,-exponent_shifts); // Scale here
#endif
						if (out_batch == (batches_of_49-1))
						{
#ifdef FIXED_POINT
							output[oo++] = (out_res.f*scalers[q]);//out_cache[cache][q]; // Scale binary accumulations
#else
							output[oo++] = out_cache[cache][q]*scalers[q];//out_cache[cache][q]; // Scale binary accumulations
#endif
						}
						else
						{
#ifdef FIXED_POINT
							output[oo++] = (out_res.f*scalers[q]);//out_cache[cache][q]; // Scale binary accumulations
#else
							output[oo++] = out_cache[cache][q]*scalers[q];//out_cache[cache][q]; // Scale binary accumulations
#endif
						}
					}
				}
				batch_pointer += (MAX_BATCH_SIZE*stride);
				out_batch_offset += (STRIPES*MAX_BATCH_SIZE);
			}
			output_offset += (STRIPES*out_size*out_size);
		}
		out_y_div_offset += (STRIPES*out_sizey*out_sizex);
	}
#ifndef ALTERA_CL
	delete img_cache;
	delete out_cache;

	//printf("max_value = %d\n",max_value);
	//printf("min_value = %d\n",min_value);
#endif

#ifdef CREATE_TEST_VECTORS
	CloseTestVectorFiles();
#endif
}

