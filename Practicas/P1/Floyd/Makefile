# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX) -D_FORCE_INLINES

all: floydSecuencial floyd1D floyd2D

floydSecuencial: floydSecuencial.cc Graph.cc Graph.h
	g++ -I./includes  -O3 floydSecuencial.cc Graph.cc -o floydSecuencial

floyd1D: floyd1D.cu Graph.cc Graph.h
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) floyd1D.cu Graph.cc -o floyd1D

floyd2D: floyd2D.cu Graph.cc Graph.h
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) floyd2D.cu Graph.cc -o floyd2D

clean:
	rm -f floydSecuencial floyd1D floyd2D *.o.
