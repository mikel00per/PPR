#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>    // std::min
using std::min;
using std::copy;
#include "floydOpenMP.h"

// Version 1D
double floyd1DOpenMP(int * M, const int N, const int P){
  int ik, kj, ij, vikj;
  int filK[N];
  int chunk = N/P;

  omp_set_dynamic(0);
  omp_set_num_threads(P);

  printf("Hay un total de %u hebras, cada se encarga de %u filas consecutivas\n", P, chunk );
  double t1 = omp_get_wtime();
  #pragma omp parallel private(ik, ij, kj, filK)
  {
    for (int k = 0; k < N; k++) {
      #pragma omp single copyprivate(filK)
      {
        for (int i = 0; i < N; i++) {
          filK[i] = M[k * N + i];
        }
      }
      #pragma omp for schedule(static, chunk)  // inicio de la región paralela, reparto estático por bloques
      for (int i = 0; i < N; i++) {
        ik = i * N + k;
        for (int j = 0; j < N; j++) {
          if (i != j && i != k && j != k) {
            kj = k * N + j;
            ij = i * N + j;
            M[ij] = min(M[ik] + filK[j], M[ij]);
          }
        }
      }
    }
  }
  double t2 = omp_get_wtime();

  return (t2-t1);
}



// Version 2D
double floyd2DOpenMP(int * M, const int N, const int P){
  int k, i, j, vikj;
  int sqrtP = sqrt(P);
  int tamBloque = N/sqrtP;

  omp_set_dynamic(0);
  omp_set_num_threads(P);

  int iLocalInicio, iLocalFinal;
  int jLocalInicio, jLocalFinal;
  iLocalInicio = iLocalFinal = jLocalInicio = jLocalFinal = 0;

  // filak y columnak declaradas en el stack
  int filak[N];
  int columnak[N];

  for(i = 0; i<N; i++){
    filak[i] = 0;
    columnak[i] = 0;
  }

  printf("Hay un total de %u hebras\n", P );
  double t1 = omp_get_wtime();

  // ponemos la construccion del bloque paralelo aqui para ahorrar sincronizacion de hebras
  #pragma omp parallel shared(M,tamBloque,columnak,filak) private(k,i,j,vikj,iLocalInicio,iLocalFinal,jLocalInicio,jLocalFinal)
  {
    int tidDSqrt = omp_get_thread_num()/sqrtP;
    int tidMSqrt = omp_get_thread_num()%sqrtP;

    iLocalInicio = tidDSqrt * tamBloque;
    iLocalFinal = (tidDSqrt+1) * tamBloque;
    jLocalInicio = tidMSqrt * tamBloque;
    jLocalFinal = (tidMSqrt+1) * tamBloque;

    for(k = 0; k<N; k++){
      int kn = k * N;

      #pragma omp barrier
      if (k >= iLocalInicio && k < iLocalFinal)
        for(i = 0; i<N; i++)
          filak[i] = M[kn + i];

      if (k >= jLocalInicio && k < jLocalFinal)
        for(i = 0; i<N; i++)
          columnak[i] = M[i * N + k];
      #pragma omp barrier

      for(i = iLocalInicio; i<iLocalFinal; i++){
        int in = i * N;
        for(j = jLocalInicio; j<jLocalFinal; j++){
          if (i != j && i != k && j != k){
            int ij = in + j;
            vikj = columnak[i] + filak[j];
            vikj = min(vikj, M[ij]);
            M[ij] = vikj;
          }
        }
      }
    }
  } // end parallel
  double t2 = omp_get_wtime();

  return (t2-t1);
}
