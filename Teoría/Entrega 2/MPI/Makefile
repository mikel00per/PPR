all: main

main: main.o
	mpicxx main.o -o main

main.o: main.cpp
	mpicxx -O3 -c main.cpp

clean:
	rm main main.o
