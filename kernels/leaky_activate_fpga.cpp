
// Type		: FPGA OpenCL kernel
// Description	: this small kernel is used to accelerate on the FPGA device the leaky activation function
// Author	: alberto scionti -- ph.d (ISMB)
// Version	: 0.1
// Last modify	: Dec 21st, 2017

#include "../kernels/fpga_channels.h"

#ifdef ALTERA_CL
__kernel
void leaky_activate_fpga (__global float* restrict inbuf, __global float* restrict outbuf, int iosize)
#else
void leaky_activate_fpga (float *inbuf, float *outbuf, int iosize)
#endif
{
	// data
	int iter;

	// code
	for (iter = 0; iter < iosize; iter++)
	{
		if (inbuf[iter] < 0.0)
		{
			outbuf[iter] = 0.1 * inbuf[iter];
		}
		else
		{
			outbuf[iter] = inbuf[iter];
		}
	}
}



