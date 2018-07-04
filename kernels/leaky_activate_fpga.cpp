
// Type		: FPGA OpenCL kernel
// Description	: this small kernel is used to accelerate on the FPGA device the leaky activation function
// Author	: alberto scionti -- ph.d (ISMB),
//			  modified by Richard Chamberlain to include variance and bias calculations
// Version	: 0.1
// Last modify	: Dec 21st, 2017

#include "basic_convolution_striped.h"

#ifdef ALTERA_CL
__kernel
void leaky_activate_fpga (__global float *restrict inbuf, __global float *restrict outbuf, int batch,int out_c,int size,
		__global float *restrict data,
		/*__global float *restrict div_sqrt_variance,
		__global float *restrict rolling_mean,
		__global float *restrict scales,
		__global float *restrict biases,*/int batch_normalised,
		 int activation)
#else
void leaky_activate_fpga (float *inbuf, float *outbuf, int batch,int out_c,int size,
						  float *data,
						  //float *div_sqrt_variance,
						  //float *rolling_mean,
						  //float *scales,
						  //float *biases,
						  int batch_normalised,
						  int activation)
#endif
{
	float div_sqrt_variance[1024];
	float rolling_mean[1024];
	float scales[1024];
	float biases[1024];
	unsigned char j = 0;
	unsigned short k = 0;
	unsigned short c = 0;
	while (j < 4)
	{
		float d = data[c++];
		if (j==0) div_sqrt_variance[k] = d;
		if (j==1) rolling_mean[k] = d;
		if (j==2) scales[k] = d;
		if (j==3) biases[k] = d;
		k = k != (out_c-1)?k+1:0;
		j = (k == 0) ? j+1:j;
	}
	int b, f, i;
	// Load inputs from common memory memory to reduce impact on resource
	for(b = 0; b < batch; ++b){
		 for(f = 0; f <out_c; f += STRIPES){
			 for (int s = 0; s < STRIPES; s++)
			 {
				 float norm;
				 float bias;
				 float leaky_activation;
				 float scale;
				 for(i = 0; i < size; ++i){
					int index = b*out_c*size + f*size + s + (i*STRIPES);
					if (batch_normalised == 1)
					{
						norm =  (inbuf[index] - rolling_mean[f+s])*div_sqrt_variance[f+s];
						scale = scales[f+s];
					}
					else
					{
						norm = inbuf[index];
						scale =1.0f;
					}
					bias = norm*scale + biases[f+s];
					float scaler = activation==FPGA_LEAKY?0.1f:1.0f;
					leaky_activation = bias < 0.0?scaler*bias:bias;
					outbuf[index] = leaky_activation;
				 }
			 }
		 }
	 }
}



