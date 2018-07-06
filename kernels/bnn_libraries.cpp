/*
 * bnn_libraries.cpp
 *
 *  Created on: Jul 3, 2018
 *      Author: rchamberlain
 *
 * BNN HDL library interface
 *
 */

#include "bnn_libraries.h"

inline int GetBit2(unsigned int val,int id)
{
	return ((val>>id)&0x1);
}

int Sum32(short input[32],unsigned int bits)
{
	bool bits2[32];
	#pragma unroll
	for (int p = 0; p < 32; p++)
		bits2[p] = GetBit2(bits,p);
	short Sum0[32];
	short Sum1[16],Sum2[8],Sum3[4];
	int Sum4[2];


	#pragma unroll
	for (int p = 0; p < 32; p++)
	{
		Sum0[p] = bits2[p]?input[p]:-input[p];
	}
	#pragma unroll
	for (int p = 0; p < 16; p++)
	{
		Sum1[p] = Sum0[p]+Sum0[p+16];
	}
	#pragma unroll
	for (int p = 0; p < 8; p++)
	{
		Sum2[p] = Sum1[p]+Sum1[p+8];
	}
	#pragma unroll
	for (int p = 0; p < 4; p++)
	{
		Sum3[p] = Sum2[p]+Sum2[p+4];
	}
	#pragma unroll
	for (int p = 0; p < 2; p++)
	{
		Sum4[p] = Sum3[p]+Sum3[p+2];
	}
	return Sum4[0] + Sum4[1];
}


#ifdef ALTERA_CL
int Sum16(short16 input,unsigned int bits)
#else
int Sum16(short input[16],unsigned int bits)
#endif
{
	bool bits2[16];
	#pragma unroll
	for (int p = 0; p < 16; p++)
		bits2[p] = GetBit2(bits,p);
	short Sum0[16];
	short Sum1[8],Sum2[4],Sum3[2];



	#pragma unroll
	for (int p = 0; p < 16; p++)
	{
		Sum0[p] = bits2[p]?input[p]:-input[p];
	}
	#pragma unroll
	for (int p = 0; p < 8; p++)
	{
		Sum1[p] = Sum0[p]+Sum0[p+8];
	}
	#pragma unroll
	for (int p = 0; p < 4; p++)
	{
		Sum2[p] = Sum1[p]+Sum1[p+4];
	}
	#pragma unroll
	for (int p = 0; p < 2; p++)
	{
		Sum3[p] = Sum2[p]+Sum2[p+2];
	}
	return (int)(Sum3[0] + Sum3[1]);
}

// Striped 32 times with input data set at 16 bits
void CountBitsHDL32x16(short input[32],unsigned int binary_coeffs[32],int results[32])
{

#ifdef USE_LIBS
	// union to convert to wide OpenCL vector types
	union {short input[32]; uint16 vec;} input_vec;
	union {short binary_coeffs[32]; ulong16 vec;} coeffs_vec;
	union {int results[32]; ulong16 vec;} results_vec;
#pragma unroll
	for (int p = 0; p < 32; p++)
	{
		input_vec.input[p] = input[p];
		coeffs_vec.binary_coeffs[p] = binary_coeffs[p];
	}
	// Call IP
	results_vec.vec = BCNNx16BITx32(coeffs_vec.vec,input_vec.vec);
#pragma unroll
	for (int p = 0; p < 32; p++)
	{
		results[p] = results_vec.results[p];
	}
#else
#pragma unroll
	for (int p = 0; p < 32; p++)
		results[p] = Sum32(input,binary_coeffs[p]);
#endif
}

#ifdef ALTERA_CL
int16 CountBitsHDL16x16(short16 input,uint16 binary_coeffs)
#else
void CountBitsHDL16x16(short input[16],unsigned int binary_coeffs[16],int results[16])
#endif
{
#ifdef ALTERA_CL
	int16 results;
#endif
#pragma unroll
	for (int p = 0; p < 16; p++)
		results[p] = Sum16(input,binary_coeffs[p]);
#ifdef ALTERA_CL
	return results;
#endif
}

// Convert input from float to integer efficiently!
short FloatToShort(float a,const int scale)
{
	#ifdef USE_LIBS
	return FloatToShortExp(a,10); // bits to shift not scale
	#else
	return (short)(a*scale);
	#endif
}

