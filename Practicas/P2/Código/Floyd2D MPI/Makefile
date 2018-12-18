
CC = g++
CXX = mpicxx
NAME = floyd
CFLAGS = -O3

MODULES = floyd2D.o Graph.o

all 		:	floyd
Graph.o		:       Graph.cc Graph.h
	$(CXX) $(CFLAGS) -c Graph.cc
	
floyd.o   :	floyd.cc
	$(CXX) $(CFLAGS) -c floyd2D.cc
	
floyd :		$(MODULES)
	$(CXX) $(CFLAGS) $(MODULES) -o $(NAME) $(EXTRALIBS)
	
clean	:
	rm -f $(MODULES)  $(NAME)
