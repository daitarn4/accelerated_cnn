GPU=0
FPGA=1
HARDWARE=1
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=1
CYGWIN=1

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples:./kernels
ifeq ($(HARDWARE),1)
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
COMMON= -Iinclude/ -Isrc/ 
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC 
#-std=c99

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
COMMON+= -I/cygdrive/d/Cuda/include
COMMON+= -I/usr/include
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
LDFLAGS+= -L/cygdrive/d/Cuda/lib/x64
LDFLAGS+= -L/cygdrive/d/Cuda/bin
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o attention.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

DEPS=
ifeq ($(FPGA),1)
COMMON+=-Ikernels/
ifeq ($(HARDWARE),1)

# OpenCL compile and link flags.
ifeq ($(CYGWIN),0)
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
endif
 

OPENCL_DIR=$(ALTERAOCLSDKROOT)/host
COMMON+= -Ikernels/common/inc/ -I$(OPENCL_DIR)/include/ 
OBJ+=opencl.o options.o darknet_kernels.o
#kernel files
OBJ+=basic_convolution_striped.o
OBJ+=basic_convolution_striped_reduced_mem.o
OBJ+=leaky_activate_fpga.o

ifeq ($(CYGWIN),1)
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
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
#DEPS = $(wildcard src/*.h) Makefile include/darknet.h
DEPS += $(wildcard src/*.h) include/darknet.h kernels/darknet_kernels.h
DEPS += $(wildcard kernels/*.h)

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)

