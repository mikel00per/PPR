#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::min;

#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string.h>
#include <time.h>
#include <string.h>
#include <sstream>

#include "cuda_runtime.h"

#include "Graph.h"

//#define blockS 16
//#define blockS 32
//define blockS 64
//#define blockS 128
//#define blockS 256
//#define blockS 512
#define blockS 1024

using namespace std;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
  if (code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Función para actualizar el kernel bidimensional.
 * @param M      Matriz
 * @param nverts numero de vertices
 * @param k      iteracciones
 */
__global__ void floyd1DKernel(int * M, const int nverts, const int k) {
  int ij = threadIdx.x + blockDim.x * blockIdx.x;
  if (ij < nverts * nverts) {
  	int Mij = M[ij];
    int i= ij / nverts;
    int j= ij - i * nverts;
    if (i != j && i != k && j != k) {
  		int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
  	}
  }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Función para ejecutar el algoritmo de floyd en la GPU usando CUDA.
 * @param h_M             Puntero al inicio de matriz.
 * @param N               Numero de filas
 * @param numBlocks       Numero de bloques que se van a usar
 * @param threadsPorBloque Número de threads por bloque.
 */
void floyd1DGPU(int *h_M, int N, int numBloques, int numThreadsBloque){
    cudaError_t err;
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(int);

    // GPU variables
    int * d_M;

    err = cudaMalloc((void **)&d_M, memSize);
  	if (err != cudaSuccess) {
  		cout << "ERROR MALLOC d_M" << endl;
  	}

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    err = cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice);
  	if (err != cudaSuccess) {
  		cout << "ERROR COPIA A GPU" << endl;
  	}

    cout << "GPU: Calculando..." << endl;
    for(int k = 0; k < N; k++){
        cout << "KERNEL: " << k << endl;
        floyd1DKernel<<< numBloques, numThreadsBloque >>> (d_M, N, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          fprintf(stderr, "Failed to launch kernel!\n");
          exit(EXIT_FAILURE);
        }
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;

    err = cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
  		cout << "ERROR COPIA A GPU" << endl;
  	}

    int i,j;
    for(i=0;i<N;i++) {
      cout << "A["<<i << ",*]= ";
      for(j=0;j<N;j++) {
        if (h_M[i*N+j]==INF)
          cout << "INF";
        else
          cout << h_M[i*N+j];
        if (j < N-1)
          cout << ",";
        else
          cout << endl;
      }
    }

    // Flush all profile data before the application exits

}

void guardarArchivo(std::string outputFile, int n, double t){
  std::ofstream archivo (outputFile.c_str(), std::ios_base::app | std::ios_base::out);
  if (archivo.is_open()){
    std::stringstream ns, ts;
    ns << n;
    ts << t;
    std::string input =  ns.str() + "\t" + ts.str() + "\n";
    archivo << input;
    archivo.close();
  }
  else
    cout << "No se puede abrir el archivo";
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Función para copiar en la matriz h_M el grafo g con tamaño N.
 * @param h_M matriz donde se van a copiar
 * @param g   grafo que se va a copiar
 * @param N   tamaño del grafo.
 */
void copiaGrafo(int * h_M, Graph g, int N){
	for(int i = 0; i<N; i++)
		for(int j = 0; j<N; j++)
			h_M[i * N + j] = g.arista(i,j);
}

void escribeGrafo(int * h_M, Graph g, int N) {
  for(int i = 0; i<N; i++)
    for(int j = 0; j<N; j++)
    g.inserta_arista(i,j, h_M[(i*N)+j]);
}

/**
 * Función principal que ejecuta todo el algoritmo Floyd.
 * @param  argc número de argumentos del programa
 * @param  argv Vector de argumentos del programa
 * @return si ha terminado bien o no la ejecución.
 */
int main(int argc, char **argv){

  if (argc != 2) {
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    return(-1);
  }

  // Get Device Information
  int devID = 0;
  CUDA_CHECK(cudaSetDevice(devID));
  CUDA_CHECK(cudaGetDevice(&devID));

  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
  if (deviceProp.computeMode == cudaComputeModeProhibited){
      cerr << "Error: La GPU no permite realizar computo ahora mismo, las hebras no pueden usar ::cudaSetDevice()." << endl;
      exit(EXIT_SUCCESS);
  }else
    cout << "GPU Device " << devID << ": \"" << deviceProp.name << "\" with compute capability " << deviceProp.major << "." << deviceProp.minor << endl << endl;

  // CPU variables
  Graph G;
  G.lee(argv[1]);
  //G.imprime();

  const unsigned int N = G.vertices;
  const unsigned int sizeMatrix = N * N;
  const unsigned int memSize = sizeMatrix * sizeof(int);
  int * h_M = (int *) malloc(memSize);
  copiaGrafo(h_M, G, N);
  cout << "Grafo copiado en matriz en la RAM" << endl ;

  int numThreadsBloque = blockS;
  int numBloques = (sizeMatrix + numThreadsBloque - 1) / numThreadsBloque;

  cout << "El blockSize es de: " << blockS << endl;
  cout << "El numBloques es de: " << numBloques << endl;
  cout << "El numThreadsBloque es de: " << numThreadsBloque << endl << endl;

  // Calc
  cout << "CPU: Mostrando resultados..." << endl;
  cout << "El Grafo con las distancias de los caminos más cortos es:" << endl << endl;

    /**
    double t1 = clock();
      floyd1DGPU(h_M, N, numBloques, numThreadsBloque);
    double Tgpu = clock();


    **/

    cudaError_t err;

    // GPU variables
    int * d_M;

    err = cudaMalloc((void **)&d_M, memSize);
    if (err != cudaSuccess) {
      cout << "ERROR MALLOC d_M" << endl;
    }

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    err = cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cout << "ERROR COPIA A GPU" << endl;
    }

    cout << "GPU: Calculando..." << endl;
    double t1 = clock();
    for(int k = 0; k < N; k++){
        floyd1DKernel<<< numBloques, numThreadsBloque >>> (d_M, N, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          fprintf(stderr, "Failed to launch kernel!\n");
          exit(EXIT_FAILURE);
        }
    }
    double Tgpu = clock();

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    err = cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      cout << "ERROR COPIA A GPU" << endl;
    }

    int i,j;
    for(i=0;i<N;i++) {
      cout << "A["<<i << ",*]= ";
      for(j=0;j<N;j++) {
        if (h_M[i*N+j]==INF)
          cout << "INF";
        else
          cout << h_M[i*N+j];
        if (j < N-1)
          cout << ",";
        else
          cout << endl;
      }
    }

    Tgpu = (Tgpu-t1)/CLOCKS_PER_SEC;
    cout << "Tiempo gastado GPU = " << Tgpu << endl << endl;

    escribeGrafo(h_M,G,N);

    G.obtenMasLargo();
    G.obtenMasCorto();
    cout << endl;



    // Guardar en el archivo los resultados
    std::string archivo = "output/floyd1D.dat";
    guardarArchivo(archivo, N, Tgpu);

    // Liberando memoria de CPU
    free(h_M);
}
