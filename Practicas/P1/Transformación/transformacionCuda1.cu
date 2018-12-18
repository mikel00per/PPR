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

//#define BLOCKSIZE 16
//#define BLOCKSIZE 32
#define BLOCKSIZE 64
//#define BLOCKSIZE 128
//#define BLOCKSIZE 256
//#define BLOCKSIZE 512
//define BLOCKSIZE 1024

using namespace std;

////////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
  if (code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void transformacionKernel(float * d_phi, float * d_phi_new, float cu, int n) {
  int i=threadIdx.x+blockDim.x*blockIdx.x+1;
  // Lax-Friedichs Stencil
  if (i<n+2)
    d_phi_new[i] = 0.5*((d_phi[i+1]+d_phi[i-1]) - cu*(d_phi[i+1]-d_phi[i-1]));
  // Impose Boundary Conditions
  if (i==1) d_phi_new[0]=d_phi_new[1];
  if (i==n+1) d_phi_new[n+2]=d_phi_new[n+1];
}

__global__ void transformacionKernel2(float * d_phi, float * d_phi_new, float cu, const int n){

  int li=threadIdx.x+1; //local index in shared memory vector
  int gi= blockDim.x*blockIdx.x+threadIdx.x+1; // global memory index
  int lstart=0; // start index in the block (left value)
  int lend=BLOCKSIZE+1; // end index in the block (right value)
  __shared__ float s_phi[BLOCKSIZE + 2]; //shared mem. vector
  float result;

  // a) Load internal points of the tile in shared memory
  if (gi<n+2) s_phi[li] = d_phi[gi];

  // b) Load the halo points of the tile in shared memory
  if (threadIdx.x == 0) // First Thread (in the current block)
    s_phi[lstart]=d_phi[gi-1];
  if (threadIdx.x == BLOCKSIZE-1) // Last Thread
    if (gi>=n+1) // Last Block
      s_phi[(n+2)%BLOCKSIZE]=d_phi[n+2];
    else
      s_phi[lend]=d_phi[gi+1];

  __syncthreads(); // Barrier Synchronization

  if (gi<n+2){
    // Lax-Friedrichs Update
    result=0.5*((s_phi[li+1]+s_phi[li-1])-cu*(s_phi[li+1]-s_phi[li-1]));
    d_phi_new[gi]=result;
  }
  // Impose Boundary Conditions
  if (gi==1) d_phi_new[0]=d_phi_new[1];
  if (gi==n+1) d_phi_new[n+2]=d_phi_new[n+1];

}

int main(int argc, char const *argv[]) {

  int Bsize, NBlocks;
  if (argc != 3){
    cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
    return(0);
  }else{
    NBlocks = atoi(argv[1]);
    Bsize= atoi(argv[2]);
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


  //* pointers to host memory */
  int *A, *B;
  const int N = Bsize * NBlocks;
  const unsigned int memSize = N * sizeof(int);

  //* Allocate arrays a, b and c on host*/
  A = (int *) malloc(memSize);
  B = (int *) malloc(memSize);

  //* Initialize arrays A and B */
  for (int i=0; i<N;i++){
    A[i]= (int) (1  -(i%100)*0.001);
    B[i]= (int) (0.5+(i%10) *0.1  );
  }

  cudaError_t err;

  int * d_A, d_B, d_C;
  err = cudaMalloc((void **)&d_A, memSize);
  if (err != cudaSuccess) {
    cout << "ERROR MALLOC d_M" << endl;
  }

  err = cudaMalloc((void **)&d_B, memSize);
  if (err != cudaSuccess) {
    cout << "ERROR MALLOC d_M" << endl;
  }

  err = cudaMalloc((void **)&d_C, memSize);
  if (err != cudaSuccess) {
    cout << "ERROR MALLOC d_M" << endl;
  }


  err = cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cout << "ERROR MEMCPY d_M" << endl;
  }

  err = cudaMemcpy(d_B, B, memSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cout << "ERROR MEMCPY d_M" << endl;
  }


  cout << "GPU: Calculando..." << endl;
  double t1=clock();
    for (size_t k = 1; k < N; k++) {
      int blocksPerGrid = ceil((float) (N+1)/BLOCKSIZE);
      transformacionKernel2<<<blocksPerGrid,BLOCKSIZE >>>(d_A, d_B,0.3,N);
    }
  double t2 = clock();

  cudaMemcpy(d_C, d_B, N, cudaMemcpyDeviceToHost);


/*
  cout<<"................................."<<endl;
  for (int k=0; k<NBlocks;k++)    cout<<"D["<<k<<"]="<<D[k]<<endl;
  cout<<"................................."<<endl<<"El valor máximo en C es:  "<<mx<<endl;
*/
  cout<<endl<<"N="<<N<<"= "<<Bsize<<"*"<<NBlocks<<"  ........  Tiempo gastado CPU= "<<t2<<endl<<endl;




  return 0;
}
