#include <iostream>
#include <vector>
#include "mpi.h"

#define MAXIMO 50
#define ITERS 10

using namespace std ;

int main(int argc, char *argv[]) {
  int numProcesos, idProceso;
  int tam;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesos);
  MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

  std::vector<double> entrada, salida;
  salida.resize(MAXIMO);
  entrada.resize(MAXIMO);

/*
    La pos 0 y 51 son las "fantasma"
*/

  for (size_t i = 0; i < entrada.size(); i++)
    entrada[i] = i+1;

  if (idProceso == 0) {
    std::cout << "Entrada = ";
    for (size_t i = 0; i < entrada.size(); i++)
      std::cout << "[" << entrada[i] << "] ";
    std::cout << endl << endl;


    tam = entrada.size();
  }

  // Todos conocen el tamaño del array
  MPI_Status estado;
  MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Cada proceso tiene su vector más dos celdas fantasmas
  std::vector<double> local;
  local.resize(entrada.size()/numProcesos+2, 0);


  for (size_t k = 0; k < ITERS; k++) {
    MPI_Scatter(&entrada[0],
      entrada.size()/numProcesos,
      MPI_DOUBLE,
      &(local[1]),
      entrada.size()/numProcesos,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
    );

    if(idProceso ==0){
      // Proceso 0:
      MPI_Send(&local[tam/2], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      MPI_Send(&local[1],     1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

      MPI_Recv(&local[0],       1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &estado);
      MPI_Recv(&local[tam/2+1], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &estado);
    }else{
      // Proceso 1:
      MPI_Send(&local[tam/2], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&local[1]    , 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

      MPI_Recv(&local[0],       1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &estado);
      MPI_Recv(&local[tam/2+1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &estado);
    }

    for (size_t i = 1; i < tam/2+1; i++) {
      local[i] = (local[i-i] - local[i] + local[i+1]) / 2;
//      cout << "Proceso[" << idProceso << "]: i = (" << local[i-i] << " - "
//                         << local[i] << " + " << local[i+i] << ") /2" << endl;
     MPI_Barrier(MPI_COMM_WORLD);

     MPI_Gather(&local[1],
       salida.size()/numProcesos,
       MPI_DOUBLE,
       &(salida[0]),
       salida.size()/numProcesos,
       MPI_DOUBLE,
       0,
       MPI_COMM_WORLD);

       entrada = salida;
    }
  }

  if (idProceso == 0) {
    std::cout << "SALIDA = ";
    for (size_t i = 0; i < entrada.size(); i++)
      std::cout << "[" << entrada[i] << "] ";
    std::cout << endl << endl;
  }

  MPI_Finalize();

/*

  double buff;
  //for (size_t k = 0; k < ITERS; k++) {
    if (idProceso == 0) {
      // Proceso 0: Envia 1 a 51 y recibe 0 de 50
      MPI_Send(&array1[1], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(&array1[0], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &estado);
    }else{
      // Proceso 1: Envia el 50 a 0 y recibe 51 de 1
      MPI_Send(&array1[tam]  , 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(&array1[tam+1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &estado);
    }

      if (idProceso == 0) {
        std::cout << " 0 Array1 = ";
        for (size_t i = 0; i < array1.size(); i++)
          std::cout << "[" << array1[i] << "] ";
        std::cout << endl << endl;
      }else{
        std::cout << " 1 Array1 = ";
        for (size_t i = 0; i < array1.size(); i++)
          std::cout << "[" << array1[i] << "] ";
        std::cout << endl << endl;
      }


/*
    for (size_t i = 0; i < entrada.size() / numProcesos; i++) {
    }
*/

//  }


/*
  // Creamos los comunicadores, en este caso el horizantal que es un vector solo
  MPI_Comm commHorizontal
  MPI_Comm_split(MPI_COMM_WORLD, idHorizontal, idProceso, &commHorizontal);
  MPI_Comm_rank(commHorizontal, &idProcesoHorizontal);


  // Creamos los bloques para la comunicación
  MPI_Datatype MPI_BLOQUE;
  salida.resize(entrada.size());

  if (idProceso == 0) {
    MPI_Type_vector(
      3,3
      entrada.size(),
      MPI_INT,
      &MPI_BLOQUE
    );

    MPI_Type_commit(&MPI_BLOQUE);
  }
*/


  return 0;
}
