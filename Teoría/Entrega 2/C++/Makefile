all: main

main: main.o
	g++ main.o -o main

main.o: main.cpp
	g++ -O3 -c main.cpp

clean:
	rm main main.o
