#include "convolutional_layer.h"
#ifdef FPGA
#include "darknet_kernels.h"
#endif
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}
   
void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] =  (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            4000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            4000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            4000000000,
            &l->bf_algo);
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef FPGA
#ifdef HARDWARE
    printf("setup layer\n");
    l.first = 1;// Load coeffs once
    // Create buffers and setup hardware if required
    cl_context context = GetFPGAContext();
    if (context == 0) // Then hardware not initialised
    {
    	host_setup(l.binary);
    	context = GetFPGAContext();
    }

    cl_int status;
    // buffers need to multiples of STRIPES
   	int ln_rounded = l.n;
	if (l.n%STRIPES)
	{
		ln_rounded += STRIPES-(l.n%STRIPES);
	}

    //l.fpga_inbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*l.w*l.h*(l.c < STRIPES?STRIPES:l.c) * sizeof(float), NULL, &status);
    
    int l_c = l.c;
    int l_n = l.n;
    l_c = (l.c%STRIPES)?l.c + (STRIPES-(l.c%STRIPES)):l.c;
    l_n = (l.n%STRIPES)?l.n + (STRIPES-(l.n%STRIPES)):l.n;
    l.fpga_outbuf  = clCreateBuffer(context, CL_MEM_READ_ONLY, 2*l.w*l.h*l_n* sizeof(float), NULL, &status);
    // binary size
    
    l.fpga_coeffbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*4*l.size*l.size*l_c*l_n / STRIPES, NULL, &status);
    l.fpga_scales_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*ln_rounded * sizeof(float), NULL, &status);
    l.fpga_biases_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*1024 * 4 * sizeof(float), NULL, &status);
    /*printf("coeff buf status = %d\n",status);
    if (l.binary)
       l.fpga_binaryscalebuf = clCreateBuffer(context, CL_MEM_READ_WRITE, l.nweights * sizeof(float), NULL, &status);
    l.fpga_div_sqrt_variance_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*ln_rounded * sizeof(float), NULL, &status);
    l.fpga_rolling_mean_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*ln_rounded * sizeof(float), NULL, &status);
    l.fpga_scales_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*ln_rounded * sizeof(float), NULL, &status);
    l.fpga_biases_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*ln_rounded * sizeof(float), NULL, &status);*/
#endif
#endif

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;
    // Calculate clocks required to process each layer!
    unsigned int clocks = l.out_w*l.out_h*l.out_c*l.c * size*size;
    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c,clocks);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}



int lcount = 0;
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    double start = what_time_is_it_now();
#ifndef FPGA
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
#endif
#ifdef FPGA
    if (l.fpga_load || l.first)
#endif
    if(l.xnor || l.binary){
    	printf("binary convolution\n");
#ifdef FPGA
    	if (l.first)
#endif
	        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        if (l.xnor)
        {
			binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
			net.input = l.binary_input;
        }
    }
#ifdef FPGA
#define BINARY_NETWORK_TEST
#endif
#ifdef BINARY_NETWORK_TEST
    if (l.binary)// && (l.stride == 2))// && (l.c==32) && (l.n == 64))
    {
        printf("fpga binary!\n");
	int c,i;
        for (c = 0; c < l.c; c++)
    	for (i = 0; i < l.w*l.w; i++)
    	{
    		//net.input[i + (c*l.w*l.w)] = ((float)rand())/RAND_MAX;//(float)c/1024.0f;
    	}
    	for (i = 0; i < l.nweights; i++)
    	{
    		//l.weights[i] = (i&0x1)?1.0f:-1.0f;
    	}

    	forward_convolution_fpga_binary_v2(l,net);
#ifdef TEST
    	unsigned int size = l.out_w*l.out_w*l.n;
    	float *temp = (float*)malloc(size*4);
	//int i;
    	printf("size = %d\n",size);
    	printf("l.outputs = %d\n",l.outputs*l.batch);
    	if (size != l.outputs*l.batch) exit(1);
    	for (i = 0; i < size; i++)
    	{
    		temp[i] = l.output[i];
    	}
        fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    	int m = l.n/l.groups;
		int k = l.size*l.size*l.c/l.groups;
		int n = l.out_w*l.out_h;
		for(i = 0; i < l.batch; ++i){
			for(j = 0; j < l.groups; ++j){
				float *a = l.weights + j*l.nweights/l.groups;
				float *b = net.workspace;
				float *c = l.output + (i*l.groups + j)*n*m;

				im2col_cpu(net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
					l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
				gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
			}
		}

		if(l.batch_normalize){
			forward_batchnorm_layer(l, net);
		  } else {
			add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
		}
		activate_array(l.output, l.outputs*l.batch, l.activation);

		size = l.out_w*l.out_w*l.n;
	int errors = 0;
    	for ( i = 0; i < size; i++)
    	{
    		int a,b;
    		a =temp[i]*1024;
    		b =l.output[i]*1024;

    		if (fabs(a-b) > 32)
    		//if (fabs(a-b) > 32)
    		{
    			printf("fpga does not match software! %d vs %d @ %d\n",a,b,i);
			errors++;
    		}
		if (errors > 32)
			exit(1);//break;
    	}


#endif
    }
    else
    {
        printf("cpu binary!\n");
        fill_cpu(l.outputs*l.batch, 0, l.output, 1);
	    int m = l.n/l.groups;
		int k = l.size*l.size*l.c/l.groups;
		int n = l.out_w*l.out_h;
		for(i = 0; i < l.batch; ++i){
			for(j = 0; j < l.groups; ++j){
				float *a = l.weights + j*l.nweights/l.groups;
				float *b = net.workspace;
				float *c = l.output + (i*l.groups + j)*n*m;

				im2col_cpu(net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
					l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
				gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
			}
		}

		if(l.batch_normalize){
			forward_batchnorm_layer(l, net);
		  } else {
			add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
		}
		activate_array(l.output, l.outputs*l.batch, l.activation);
    }
#else

#ifdef FPGA
    //if (/*(l.w == 13) ||*/ (l.w == 26)// || (l.w == 52) || (l.w == 104) || (l.w == 208) || (l.w == 416))
//    	&& (lcount < 17))
    	//if ((l.w == l.w))
		{
    		int k  = 0;
    		for (int j = 0; j < l.c; j++)
    		for (int i = 0; i <l.w*l.w;i++)
    		{
        		//net.input[k++] = j;
        	}

    		k  = 0;
    		for (int j = 0; j < l.c; j++)
    		for (int i = 0; i <l.n*l.size*l.size;i++)
    		{
        		//sl.weights[k++] = 1;
        	}


    		forward_convolution_fpga(l,net);

   // 		for (int i = 0; i < l.w*l.w*l.c;i++)
    //			if (net.input[i] != 1)
    	//			printf("Corrupted\n");
    		//for (int i = 0; i <l.c*l.n*l.size*l.size;i++)
    			//if(l.weights[i] != 1)
    				//printf("Corrupted\n");

    		k  = 0;
    		for (int j = 0; j < l.c; j++)
    		for (int i = 0; i <l.w*l.w;i++)
    		{
        		//net.input[k++] = j;
        	}
    		for (int j = 0; j < l.c; j++)
    		for (int i = 0; i <l.n*l.size*l.size;i++)
    		{
        		//l.weights[k++] = j;
        	}
    		printf("lcount = %d\n",lcount);
        	lcount++;
        	// Save fpga to compare with gold to work out what is wrong
        	unsigned int size = l.out_w*l.out_w*l.n;
        	float *temp = (float*)malloc(size*4);
        	for (int i = 0; i < size; i++)
        	{
        		temp[i] = l.output[i];
        		l.output[i] = 0;
        	}

            {
        		int m = l.n/l.groups;
        		int k = l.size*l.size*l.c/l.groups;
        		int n = l.out_w*l.out_h;
        		for(i = 0; i < l.batch; ++i){
        			for(j = 0; j < l.groups; ++j){
        				float *a = l.weights + j*l.nweights/l.groups;
        				float *b = net.workspace;
        				float *c = l.output + (i*l.groups + j)*n*m;

        				im2col_cpu(net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
        					l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
        				gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        			}
        		}
        		if(l.batch_normalize){
        			forward_batchnorm_layer(l, net);
        		  } else {
        			add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
        		}
        		activate_array(l.output, l.outputs*l.batch, l.activation);
            }
            int errors  = 0;
        	for (int i = 0; i < size; i++)
        	{
        		if (fabs(l.output[i]) > 0.0001)
        		if ((temp[i]/l.output[i]) > 1.01)
        		{
        			if (errors < 5)
        			printf("%d = %f - %f\n",i,temp[i],l.output[i]);
        			errors++;
        		}
        	}
            free(temp);
		}
#else
    //else
	printf("standard\n");
    {
		int m = l.n/l.groups;
		int k = l.size*l.size*l.c/l.groups;
		int n = l.out_w*l.out_h;
		for(i = 0; i < l.batch; ++i){
			for(j = 0; j < l.groups; ++j){
				float *a = l.weights + j*l.nweights/l.groups;
				float *b = net.workspace;
				float *c = l.output + (i*l.groups + j)*n*m;

				im2col_cpu(net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
					l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
				gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
			}
		}
		if(l.batch_normalize){
			forward_batchnorm_layer(l, net);
		  } else {
			add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
		}
		activate_array(l.output, l.outputs*l.batch, l.activation);
    }

#endif
#endif


    if(l.binary || l.xnor) swap_binary(&l);
    double end = what_time_is_it_now();
    printf("convolution time  = %f\n",start-end);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im = net.input+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                    l.size, l.stride, l.pad, b);
            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if(net.delta){
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, 
                    l.pad, net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w);
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

