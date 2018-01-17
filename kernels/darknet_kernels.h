/*
 * darnet_kernels.h
 *
 *  Created on: Jan 2, 2018
 *      Author: rchamberlain
 */

#ifndef KERNELS_DARKNET_KERNELS_H_
#define KERNELS_DARKNET_KERNELS_H_

#ifndef ALTERA_CL
#include <stdio.h>
#include <malloc.h>
#endif
#include "basic_convolution_striped_reduced_mem.h"
#include "basic_convolution_striped.h"
#ifdef __cplusplus
extern "C" {
#endif
void host_setup();
void HardwareRunConvolution(float *input, float *coeffs, float *output,
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
					int y_div);

#ifdef __cplusplus
}
#endif


#endif /* KERNELS_DARKNET_KERNELS_H_ */
