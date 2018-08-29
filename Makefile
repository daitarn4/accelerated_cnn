GPU=0
FPGA=1
HARDWARE=0
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=1
CYGWIN=1
TEST=1

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

ARCH= -gencode arch=compute_52,code=[sm_52,compute_52]

#ARCH= -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_37,code=\"sm_37,compute_37\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" -gencode=arch=compute_52,code=\"sm_52,compute_52\" -gencode=arch=compute_60,code=\"sm_60,compute_60\" -gencode=arch=compute_70,code=\"sm_70,compute_70\"

VPATH=./src/:./examples:./kernels
ifeq ($(FPGA),1)
VPATH+=:./kernels/common/src/AOCLUtils
endif
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CXX=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ -I/usr/include/ 
CFLAGS= -fPIC
#CFLAGS=-Wall -Wfatal-errors -fPIC 
#-std=c99

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(TEST),1)
CFLAGS+=-DTEST
endif

CFLAGS+=-std=c++11
ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV -ID:/darknet/darknet-master/opencv/build/include
#LDFLAGS+= `pkg-config --libs opencv` -LD:/darknet/darknet-master/opencv/build/x64/vc15/lib -lopencv_world340
LDFLAGS+=-LD:/darknet/darknet-master/opencv/build/x64/vc15/lib -lopencv_world340
#COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
COMMON+= -I/cygdrive/d/Cuda/include
COMMON+= -I/usr/include
CFLAGS+= -DGPU 
LDFLAGS+= -lcuda -ldl -lcudart -lcublas -lcublas_device -lcurand -lcudart_static -lnvrtc -lcusolver
LDFLAGS+= -L/cygdrive/d/Cuda/lib/x64
LDFLAGS+= -L/cygdrive/d/Cuda/bin
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

ifeq ($(FPGA),1)
COMMON+=-Ikernels/ -DFPGA
#kernel files
OBJ+=basic_convolution_striped.o
OBJ+=basic_convolution_striped_reduced_mem.o
#OBJ+=conv_binary_fpga.o
OBJ+=conv_binary_fpga_v9.o
OBJ+=bnn_libraries.o
OBJ+=leaky_activate_fpga.o
OBJ+=max_pooling.o
OBJ+=average_pool.o
OBJ+=route_layer_fpga.o
OBJ+=shortcut_layer_fpga.o
OBJ+= darknet_kernels_v3.o
OBJ+= yolo_layer_fpga.o
OBJ+= upsample_x2_fpga.o
COMMON+= -Ikernels/common/inc/ -I$(OPENCL_DIR)/include/
OBJ+=opencl.o options.o 
 

ifeq ($(HARDWARE),1)
COMMON+= -DHARDWARE
endif

ifeq ($(CYGWIN),1)
COMMON+= -DCYGWIN
endif

# OpenCL compile and link flags.
ifeq ($(CYGWIN),0)
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
endif
 

OPENCL_DIR=$(ALTERAOCLSDKROOT)/host

ifeq ($(CYGWIN),1)

#for cuda?
CPFLAGS+= -ID:/cygwin/usr/include 
LDFLAGS+= -L$(OPENCL_DIR)/windows64/lib/
#LDFLAGS+= -L$(OPENCL_DIR)/linux64/lib/
LDFLAGS+= -lOpenCL
LDFLAGS+= -lstdc++ 
LDFLAGS+= -lacl_emulator_kernel_rt
#LDFLAGS+= -lacl_hostxml
LDFLAGS+= -lalteracl
LDFLAGS+= -lalterahalmmd
LDFLAGS+= -llibelf
endif

CFLAGS+=$(AOCL_COMPILE_CONFIG)
LDFLAGS+=$(AOCL_LINK_CONFIG)
LDFLAGS+=-L$(ALTERAOCLSDKROOT)/board/nalla_pcie/linux64/lib

#LDFLAGS+=-lnalla_pcie_mmd.so
LDFLAGS+= -lstdc++ 
endif


EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
#DEPS = $(wildcard src/*.h) Makefile include/darknet.h
DEPS += $(wildcard src/*.h) include/darknet.h kernels/darknet_kernels.h
DEPS += $(wildcard kernels/*.h)

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#all: obj results $(SLIB) $(ALIB) $(EXEC)
all: obj backup results $(SLIB) $(ALIB) $(EXEC)
$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@	
	
$(OBJDIR)%.o: %.cpp
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu 
	$(NVCC) $(ARCH) $(COMMON) -D_hypotf=hypotf -DUSING_NVCC --compiler-options "$(CFLAGS)" -c $< -o $@

#$(NVCC) $(ARCH) $(COMMON) -D__CUDACC__ -D_MATH_H -DWIN32 -D_MBCS -D_MSC_VER=1911 -DUSING_NVCC --compile -lm -lcuda -lcudart -lcublas -lcurand -lnvrtc -cudart static --machine 64 -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)

