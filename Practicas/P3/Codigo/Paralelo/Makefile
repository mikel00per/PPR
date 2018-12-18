all: bbparalelo


bbparalelo: bbparalelo.o libbb.o
	mpicxx -O3 -m64 bbparalelo.o libbb.o -o bbparalelo

bbparalelo.o: bbparalelo.cc
	mpicxx -O3 -m64 -c bbparalelo.cc


libbb.o: libbb.cc libbb.h
	mpicxx -O3 -m64 -c  libbb.cc


clean:
	/bin/rm -f *.o bbparalelo
