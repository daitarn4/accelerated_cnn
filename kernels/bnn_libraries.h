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

// Local functions

void CountBitsHDL32x16(short input[32],unsigned int binary_coeffs[32],int results[32]);

void CountBitsHDL16x16(short input[16],unsigned int binary_coeffs[16],int results[16]);

// Scale input

short FloatToShort(float a,const int scale);


#endif /* KERNELS_BNN_LIBRARIES_H_ */
