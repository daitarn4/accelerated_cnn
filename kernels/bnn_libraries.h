/*
 * bnn_libraries.h
 *
 *  Created on: Jul 3, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_BNN_LIBRARIES_H_
#define KERNELS_BNN_LIBRARIES_H_

// HDL functions
#ifdef USE_LIBS
#include "bnn_hdl.h"
#endif

int GetBit2(unsigned int val,int id);
// Local functions


void CountBitsHDL32x16(short input[32],unsigned int binary_coeffs[32],int results[32]);

#ifdef ALTERA_CL
int16 CountBitsHDL16x16(short16 input,uint16 binary_coeffs);
#else
void CountBitsHDL16x16(short input[16],unsigned int binary_coeffs[16],int results[16]);
#endif

// Scale input

short FloatToShort(float a,const short power);

// Test vector generation
#ifndef ALTERA_CL
#define CREATE_TEST_VECTORS
#endif
#ifdef CREATE_TEST_VECTORS
#include <stdio.h>

void SetupTestVectorFiles();
void CloseTestVectorFiles();
void StoreVector(FILE *file,unsigned int bits,unsigned char *c,unsigned int bit_size);
#endif

#endif /* KERNELS_BNN_LIBRARIES_H_ */
