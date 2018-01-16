#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Added to include FPGA OpenCL kernel
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

extern static const size_t work_group_size;  
// OpenCL runtime configuration
extern static cl_platform_id platform;
extern static cl_device_id device;
extern static cl_context context;
extern static cl_command_queue queue;
extern static cl_kernel kernel;
extern static cl_program program;



char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float* x, const int n, const ACTIVATION a)
{
	// data
	cl_int status;
    	int i;

	// code
	if (a == LEAKY)
	{
		// Call the FPGA kernel
		fprintf(stdout, "\n\n*** [FPGA] Launching the kernel...\n\n");
  		// Configure work set over which the kernel will execute
  		size_t wgSize[3] = {work_group_size, 1, 1};
  		size_t gSize[3] = {work_group_size, 1, 1};
  		// Create input variables to pass input-output data with the kernel
		cl_mem inbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, no_words * sizeof(int), NULL, NULL);
		cl_mem outbuf = clCreateBuffer(context, CL_MEM_READ_ONLY, no_words * sizeof(int), NULL, NULL);
		// Write input to the kernel
		clEnqueueWriteBuffer(queue, inbuf, CL_TRUE, 0, sizeof(float) * n, &x, 0, NULL, NULL);
		//Set arguements
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &inbuf);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &outbuf);
		clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
		// Launch the kernel
  		status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
  		checkError(status, "Failed to launch kernel");
		// Reading output
		clEnqueueReadBuffer(queue, outbuf, CL_TRUE, 0, sizeof(float) * n, &x, 0, NULL, NULL);	
		// Completed
		status = clFinish(queue);
		checkError(status, "Failed to finish");
		fprintf(stdout, "\n\n*** [FPGA] Kernel execution completed\n\n");
	}
	else
	{
		// perform the activation using host code
    		for(i = 0; i < n; ++i)
		{
        		x[i] = activate(x[i], a);
    		}
	}
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

