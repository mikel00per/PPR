
CC = g++
#  mpicxx
NAME1 = floyd1D
NAME2 = floyd2D

CFLAGS = -O3 -fopenmp

COMMONS = Graph.o floydOpenMP.o

MODULES1D = floyd1D.o $(COMMONS)
MODULES2D = floyd2D.o $(COMMONS)

all 		:	floyd1D floyd2D

Graph.o		:       Graph.cc Graph.h
	$(CC) $(CFLAGS) -c Graph.cc

floydOpenMP.o		:  floydOpenMP.cpp floydOpenMP.h
	$(CC) $(CFLAGS) -c floydOpenMP.cpp

floyd1D.o   :	floyd1D.cpp
	$(CC) $(CFLAGS) -c floyd1D.cpp

floyd2D.o   :	floyd2D.cpp
	$(CC) $(CFLAGS) -c floyd2D.cpp

floyd1D :		$(MODULES1D)
	$(CC) $(CFLAGS) $(MODULES1D) -o $(NAME1) $(EXTRALIBS)

floyd2D :		$(MODULES2D)
	$(CC) $(CFLAGS) $(MODULES2D) -o $(NAME2) $(EXTRALIBS)

clean	:
	rm -f $(MODULES1D) $(MODULES2D)  $(NAME1) $(NAME2)
