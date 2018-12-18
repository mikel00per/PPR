#Makefile

OBJ = transformacion
SRC = transformacion.cc

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


all: transformacionSecuencial transformacionCuda1


transformacionSecuencial: transformacionSecuencial.cc
	g++ -O3 transformacionSecuencial.cc -o transformacionSecuencial

transformacionCuda1: transformacionCuda1.cu
	nvcc -O3 -m64 $(GENCODE_FLAGS) transformacionCuda1.cu -o transformacionCuda1

clean:
	rm $(OBJ) *.o
