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
#include "CL/opencl.h"
#endif
#include "conv_binary_fpga.h"
#define V4
#ifdef V3
#include "conv_binary_fpga_v3.h"
#endif
#ifdef V4
#include "conv_binary_fpga_v8.h"
#endif
#include "basic_convolution_striped_reduced_mem.h"
#include "basic_convolution_striped_binary.h"
#include "basic_convolution_striped.h"
#include "maxpool_layer.h"
#include "route_layer.h"
#include "route_layer_fpga.h"
#include "shortcut_layer_fpga.h"
#include "upsample_x2_fpga.h"
#ifdef __cplusplus
extern "C" {
#endif

cl_context GetFPGAContext();

void host_setup(int binary);

void HardwareRunConvolution(
		float *input, float *coeffs, float *output,
		int ln_rounded,
		convolutional_layer l);

// Binary coeffs 32 per unsigned int
void HardwareRunConvolutionBinary(
		float *input, unsigned int *coeffs, float *output,
		float *scales,
		int ln_rounded,
		convolutional_layer l,
		int sub_block_size_x,
		int sub_block_size_y,
		int divx,
		int	divy,
		network net);



void forward_convolution_fpga(convolutional_layer l, network net);
void forward_convolution_fpga_binary(convolutional_layer l, network net);
void forward_convolution_fpga_binary_v2(convolutional_layer l, network net);
void activate_array_fpga(float* x, const int n, const ACTIVATION a);
void forward_maxpool_layer_fpga(const maxpool_layer l, network net);
void forward_avgpool_layer_fpga(const maxpool_layer l, network net);
void forward_route_layer_fpga(const route_layer l, network net);
void forward_shortcut_layer_fpga(const route_layer l, network net);
void forward_upsample_layer_fpga(const layer l, network net);
void forward_yolo_layer_fpga(const maxpool_layer l, network net);

#ifdef __cplusplus
}
#endif


#endif /* KERNELS_DARKNET_KERNELS_H_ */
