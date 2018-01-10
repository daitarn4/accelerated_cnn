#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// ================
// 	FPGA
// ================
// Added to manage FPGA kernels
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

#define True			1
#define False			0
#define STRING_BUFFER_LEN	1024


// ================
// 	FPGA
// ================
// Runtime constants
// Used to define the work set over which this kernel will execute:
// it represents the number of threads running (we use only 1 thread)
static const size_t work_group_size = 1;  
// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;
// Function prototypes
bool init();
void cleanup();
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name);
static void device_info_string(cl_device_id device, cl_device_info param, const char* name);
static void display_device_info(cl_device_id device);




// Function prototypes
extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_attention(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_segmenter(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);
extern void run_lsd(int argc, char **argv);


// FPGA function for initialization of the board
bool init(void) 
{
	// data
  	cl_int status;
	char char_buffer[STRING_BUFFER_LEN];
	scoped_array<cl_device_id> devices;
  	cl_uint num_devices;
	
	// code	
	if(!setCwdToExeDir()) 
	{
    		return false;
  	}
 	// Get the OpenCL platform.
  	platform = findPlatform("Intel");
  	if(platform == NULL) 
	{
    		fprintf(stdout, "ERROR: Unable to find Intel FPGA OpenCL platform.\n");
    		return false;
  	}
	// User-visible output - Platform information
  	{
		fprintf(stdout, "==========================\n");
     		fprintf(stdout, "Querying platform for info:\n");
    		fprintf(stdout, "==========================\n");
    		clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    		fprintf(stdout, "%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    		fprintf(stdout, "%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    		clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    		fprintf(stdout, "%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  	}
  	// Query the available OpenCL devices.
  	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	// We'll just use the first device.
  	device = devices[0];
	// Display some device information.
  	display_device_info(device);
	// Create the context.
  	context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  	checkError(status, "Failed to create context");
	// Create the command queue.
  	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  	checkError(status, "Failed to create command queue");
	// Create the program.
  	std::string binary_file = getBoardBinaryFile("../bitstream/leaky_activate_fpga", device);
  	fprintf(stdout, "Using AOCX: %s\n", binary_file.c_str());
  	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  	// Build the program that was just created.
  	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  	checkError(status, "Failed to build program");
	//
  	// Create the kernel - name passed in here must match kernel name in the
  	// original CL file, that was compiled into an AOCX file using the AOC tool:
	// Kernel name, as defined in the CL file
  	const char *kernel_name = "leaky_activate_fpga";
  	kernel = clCreateKernel(program, kernel_name, &status);
  	checkError(status, "Failed to create kernel");
	//
  	return true;
}

// Clean up function for the FPGA board
void cleanup() 
{
	// Code
  	if(kernel) 
	{
    		clReleaseKernel(kernel);  
  	}
  	if(program) 
	{
    		clReleaseProgram(program);
  	}
  	if(queue) 
	{
    		clReleaseCommandQueue(queue);
  	}
  	if(context) 
	{
    		clReleaseContext(context);
  	}
}

// Other helper functions
// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) 
{
	// Data
   	cl_ulong a;
   
	// Code
	clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   	fprintf(stdout, "%-40s = %lu\n", name, a);
}

static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) 
{
	// Data
   	cl_uint a;

	// Code   
	clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   	fprintf(stdout, "%-40s = %u\n", name, a);
}

static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) 
{
	// Data
   	cl_bool a;

	// Code
   	clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   	fprintf(stdout, "%-40s = %s\n", name, (a?"true":"false"));
}

static void device_info_string( cl_device_id device, cl_device_info param, const char* name) 
{
	// Data
   	char a[STRING_BUFFER_LEN]; 
   
	// Code
	clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   	fprintf(stdout, "%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device) 
{
	// Code
   	fprintf(stdout, "==========================\n");
   	fprintf(stdout, "Querying device for info: \n");
   	fprintf(stdout, "==========================\n");
   	device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   	device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   	device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   	device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   	device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   	device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   	device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  	device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   	device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   	device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   	device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   	device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   	device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
   	device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
  	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");
   	{
      		cl_command_queue_properties ccp;
      		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      		fprintf(stdout, "%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      		fprintf(stdout, "%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   	}
}



// ========================================
// Other function from Yolo-architecture
// ========================================
void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    network *sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];   
    load_weights(sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];   
        load_weights(net, weightfile);
        for(j = 0; j < net->n; ++j){
            layer l = net->layers[j];
            layer out = sum->layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net->n; ++j){
        layer l = sum->layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

long numops(network *net)
{
    int i;
    long ops = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
        } else if (l.type == RNN){
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        } else if (l.type == GRU){
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        } else if (l.type == LSTM){
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    return ops;
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network *net = parse_network_cfg(cfgfile);
    set_batch_network(net, 1);
    int i;
    double time=what_time_is_it_now();
    image im = make_image(net->w, net->h, net->c*net->batch);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = what_time_is_it_now() - time;
    long ops = numops(net);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
    printf("FLOPS: %.2f Bn\n", (float)ops/1000000000.*tics/t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    long ops = numops(net);
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    int oldn = net->layers[net->n - 2].n;
    int c = net->layers[net->n - 2].c;
    scal_cpu(oldn*c, .1, net->layers[net->n - 2].weights, 1);
    scal_cpu(oldn, 0, net->layers[net->n - 2].biases, 1);
    net->layers[net->n - 2].n = 11921;
    net->layers[net->n - 2].biases += 5;
    net->layers[net->n - 2].weights += 5*c;
    if(weightfile){
        load_weights(net, weightfile);
    }
    net->layers[net->n - 2].biases -= 5;
    net->layers[net->n - 2].weights -= 5*c;
    net->layers[net->n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net->layers[net->n - 2];
    copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
    copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
    *net->seen = 0;
    save_weights(net, outfile);
}

void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(net, weightfile, 0, net->n);
        load_weights_upto(net, weightfile, l, net->n);
    }
    *net->seen = 0;
    save_weights_upto(net, outfile, net->n);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    save_weights_upto(net, outfile, max);
}

void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = calloc(n, sizeof(float));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net->layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net->layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net->layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if ((l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net->layers[i].batch_normalize=0;
        }
    }
    save_weights(net, outfile);
}

void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{
    network *net = load_network(cfgfile, weightfile, 0);
    image *ims = get_weights(net->layers[0]);
    int n = net->layers[0].n;
    int z;
    for(z = 0; z < num; ++z){
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for(i = 0; i < 100; ++i){
            image r = copy_image(ims[rand()%n]);
            rotate_image_cw(r, rand()%4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand()%(w-r.w);
            int dy = rand()%(h-r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}

void visualize(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    visualize_network(net);
#ifdef OPENCV
    cvWaitKey(0);
#endif
}


// =================================
// 	    MAIN FUNCTION
// =================================
int main(int argc, char **argv)
{
	// Code
    	//test_resize("data/bad.jpg");
    	//test_box();
    	//test_convolutional_layer();
    	if(argc < 2)
	{
        	fprintf(stderr, "usage: %s <function>\n", argv[0]);
        	return 0;
    	}
    	gpu_index = find_int_arg(argc, argv, "-i", 0);
    	if(find_arg(argc, argv, "-nogpu")) 
	{
        	gpu_index = -1;
    	}
	
	// ====
	// Initialize the FPGA kernel 
	// ====
	if(!init()) 
	{
    		return (-1);
  	}
	// ====	
	

	#ifndef GPU
    	gpu_index = -1;
	#else
    	if(gpu_index >= 0)
	{
        	cuda_set_device(gpu_index);
    	}
	#endif
    	if (0 == strcmp(argv[1], "average"))
	{
        	average(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "yolo"))
	{
        	run_yolo(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "super"))
	{
        	run_super(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "lsd"))
	{
        	run_lsd(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "detector"))
	{
        	run_detector(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "detect"))
	{
        	float thresh = find_float_arg(argc, argv, "-thresh", .24);
        	char *filename = (argc > 4) ? argv[4]: 0;
        	char *outfile = find_char_arg(argc, argv, "-out", 0);
        	int fullscreen = find_arg(argc, argv, "-fullscreen");
        	test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    	} 
	else if (0 == strcmp(argv[1], "cifar"))
	{
 	       run_cifar(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "go"))
	{
        	run_go(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "rnn"))
	{
        	run_char_rnn(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "coco"))
	{
        	run_coco(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "classify"))
	{
        	predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    	} 
	else if (0 == strcmp(argv[1], "classifier"))
	{
        	run_classifier(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "attention"))
	{
        	run_attention(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "regressor"))
	{
        	run_regressor(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "segmenter"))
	{
        	run_segmenter(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "art"))
	{
        	run_art(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "tag"))
	{
        	run_tag(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "3d"))
	{
        	composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    	} 
	else if (0 == strcmp(argv[1], "test"))
	{
        	test_resize(argv[2]);
    	} 
	else if (0 == strcmp(argv[1], "captcha"))
	{
        	run_captcha(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "nightmare"))
	{
  	      	run_nightmare(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "rgbgr"))
	{
        	rgbgr_net(argv[2], argv[3], argv[4]);
    	} 
	else if (0 == strcmp(argv[1], "reset"))
	{
        	reset_normalize_net(argv[2], argv[3], argv[4]);
    	} 
	else if (0 == strcmp(argv[1], "denormalize"))
	{
        	denormalize_net(argv[2], argv[3], argv[4]);
    	} 
	else if (0 == strcmp(argv[1], "statistics"))
	{
        	statistics_net(argv[2], argv[3]);
    	} 
	else if (0 == strcmp(argv[1], "normalize"))
	{
        	normalize_net(argv[2], argv[3], argv[4]);
    	} 
	else if (0 == strcmp(argv[1], "rescale"))
	{
        	rescale_net(argv[2], argv[3], argv[4]);
    	} 
	else if (0 == strcmp(argv[1], "ops"))
	{
        	operations(argv[2]);
    	} 
	else if (0 == strcmp(argv[1], "speed"))
	{
        	speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    	} 
	else if (0 == strcmp(argv[1], "oneoff"))
	{
        	oneoff(argv[2], argv[3], argv[4]);
    	} 
	else if (0 == strcmp(argv[1], "oneoff2"))
	{
        	oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    	} 
	else if (0 == strcmp(argv[1], "partial"))
	{
        	partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    	} 
	else if (0 == strcmp(argv[1], "average"))
	{
        	average(argc, argv);
    	} 
	else if (0 == strcmp(argv[1], "visualize"))
	{
        	visualize(argv[2], (argc > 3) ? argv[3] : 0);
    	} 
	else if (0 == strcmp(argv[1], "mkimg"))
	{
        	mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    	} 
	else if (0 == strcmp(argv[1], "imtest"))
	{
        	test_resize(argv[2]);
    	} 
	else 
	{
        	fprintf(stderr, "Not an option: %s\n", argv[1]);
    	}

	// ====
	// Cleanup the FPGA kernel 
	// ====
  	cleanup();
	// ====

	return 0;
}

