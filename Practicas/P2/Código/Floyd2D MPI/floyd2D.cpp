#include <iostream>
#include <fstream>
#include <string.h>
#include <cmath>
#include "Graph.h"
#include "mpi.h"

#define COUT false

using namespace std;

void guardaEnArchivo(int n, double t){
  ofstream archivo ("output/floyd2D.dat" , ios_base::app | ios_base::out);
  if (archivo.is_open()){
    archivo << to_string(n) + "\t" + to_string(t) + "\n";
    archivo.close();
  }else
    cout << "No se puede abrir el archivo";
}

int main (int argc, char *argv[]){
  int numeroProcesos, idProceso;
  int nverts, nfilas, *ptrInicioMatriz = NULL;
  Graph * G;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesos);
  MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

  if (argc != 2){
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    return EXIT_FAILURE;
  }

  if(idProceso == 0){
    G = new Graph();
    G->lee(argv[1]);
    cout << "EL Grafo de entrada es:"<<endl;
    G->imprime();

    nverts = G->vertices;
    nfilas = nverts / numeroProcesos;
    ptrInicioMatriz = G->getPtrMatriz();
  }

  /*
      1. Le envio a todos el num de vertices por fila/columnak
         y el número de filas/columnas de las que se encarga cada proceso
  */
  MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nfilas, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(nverts%numeroProcesos != 0){
    cout << "P"<< idProceso<<" -> El numero de vertices no es divisible entre el numero de procesos" << endl;
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

/*
    2. Calculamos el tamaño del que se encarga cada bloque al mismo tiempo
       creamos los comunicadores para enviar los datos a los procesos bloque
       se encarguen de su correspondiente parte.
       Ya que:

        P0      |    P1
                |
       -----------------
        P2      |     P3
                |

      de esta forma para saber a que proceso nos comunicamos creamos una
      red de comunicación con idHorizontal  y idVertical
      de forma que cuando envie y reciba datos sepamos a donde enviarlos:

              H   |   V		|		iI	jI	 |	  iF	jF
        -------------------	|	-----------------------
        P0    0   |   0		|		  0	0			B	B
        P1    0   |   1		|	   	0	B			B	2B
        P2    1   |   0		|	   	B	0			2B B
        P3    1   |   1		|	   	B	B			2B	2B
*/
  int sqrtP = sqrt(numeroProcesos),
      tamBloque = nverts / sqrtP;

  int idHorizontal = idProceso / sqrtP,
      idVertical = idProceso % sqrtP,
      idProcesoHorizontal,
      idProcesoVertical;

  MPI_Comm commHorizontal, commVertical;

  // Creamos los comunicadores, los procesos con el mismo idHorizontal
  // entraran en el mismo comunicador, igual para idVertical y btenemos el
  // nuevo rango asignado dentro de commHorizontal y commVertical
  MPI_Comm_split(MPI_COMM_WORLD, idHorizontal, idProceso, &commHorizontal);
  MPI_Comm_split(MPI_COMM_WORLD, idVertical, idProceso, &commVertical);
  MPI_Comm_rank(commHorizontal, &idProcesoHorizontal);
  MPI_Comm_rank(commVertical, &idProcesoVertical);


  /*
      3. El proceso 0 crea el tipo vector que será un bloque del que se
      encargará cada hebra. Y se lo hacemos saber a todos.
      Tras esto empaqueto todo el trabajo de los procesos de la forma que quiero
  */

  MPI_Datatype MPI_BLOQUE;
  int bufferSalida[nverts*nverts];
  int filaSubmatriz, columnaSubmatriz, comienzo;

  if (idProceso == 0){
    MPI_Type_vector(tamBloque,tamBloque,nverts,MPI_INT,&MPI_BLOQUE);
    MPI_Type_commit(&MPI_BLOQUE);

    int posActualBuffer = 0;

    // Empaqueta bloque a bloque en el buffer de envio
    for (int i = 0; i < numeroProcesos; i++){
      filaSubmatriz = i / sqrtP;
      columnaSubmatriz = i % sqrtP;
      comienzo=(columnaSubmatriz*tamBloque)+(filaSubmatriz*tamBloque*tamBloque*sqrtP);

      MPI_Pack(G->getPtrMatriz()+comienzo, 1, MPI_BLOQUE, bufferSalida,
              sizeof(int) * nverts * nverts, &posActualBuffer, MPI_COMM_WORLD);
    }

    MPI_Type_free(&MPI_BLOQUE);
  }

  // Me creo una subMatriz local pequeña a cada proceso y reparto todo lo que
  // está en el buffer entre los procesos. Como antes hemos definido su tamaño
  // se repartirán de la forma que queremos.
  //
  int subMatriz[tamBloque][tamBloque];

  MPI_Scatter(bufferSalida,sizeof(int)*tamBloque * tamBloque,MPI_PACKED,
              subMatriz,tamBloque * tamBloque,MPI_INT,0,MPI_COMM_WORLD);

  // Delimitamos hasta donde va a llegar cada rpoceso en función de su id
  // y al bloque que tiene asignado.
  int i, j, k, vikj, iGlobal, jGlobal,
 /*
             H   |   V		|		iI	jI	 |	  iF	jF
      -------------------	|	-----------------------
      P0    0   |   0		|		0	0			B	B
      P1    0   |   1		|		0	B			B	2B
      P2    1   |   0		|		B	0			2B B
      P3    1   |   1		|		B	B			2B	2B

 */
      iLocalInicio = idHorizontal * tamBloque,
      iLocalFinal = (idHorizontal + 1) * tamBloque,
      jLocalInicio = idVertical * tamBloque,
      jLocalFinal = (idVertical + 1) * tamBloque,
      idProcesoBloqueK = 0,
      indicePartidaFilaK = 0;

  // Antes de de ponerme a hacer el algortimo me creo unas filas y columnas
  // k para, tener una copia de las submatrices que realizaremos en cada
  // iteracciçon
  int * filak = new int[tamBloque], * columnak = new int[tamBloque];

  for(i = 0; i<tamBloque; i++){
    filak[i] = 0;
    columnak[i] = 0;
  }

  double t = MPI_Wtime();
    for(k = 0; k<nverts; k++){
      idProcesoBloqueK = k / tamBloque;
      indicePartidaFilaK = k % tamBloque;

      // Copiamos la fila /coluna de cada cada proceso a
      if (k >= iLocalInicio && k < iLocalFinal)
        copy(subMatriz[indicePartidaFilaK], subMatriz[indicePartidaFilaK] + tamBloque, filak);

      if (k >= jLocalInicio && k < jLocalFinal)
        for (i = 0; i < tamBloque; i++)
          columnak[i] = subMatriz[i][indicePartidaFilaK];

      MPI_Barrier(MPI_COMM_WORLD);

      MPI_Bcast(filak, tamBloque, MPI_INT, idProcesoBloqueK, commVertical);
      MPI_Bcast(columnak, tamBloque, MPI_INT, idProcesoBloqueK, commHorizontal);

      for(i = 0; i<tamBloque; i++){
        iGlobal = iLocalInicio + i;
        for(j = 0; j<tamBloque; j++){
          jGlobal = jLocalInicio + j;
          if (iGlobal != jGlobal && iGlobal != k && jGlobal != k){ // evitar diagonal.
            vikj = columnak[i] + filak[j];
            vikj = min(vikj, subMatriz[i][j]);
            subMatriz[i][j] = vikj;
          }
        }
      }
    }
  t = MPI_Wtime()-t;

  MPI_Barrier(MPI_COMM_WORLD);

  // Cuando acabamos ponemos un barrier en common para parar todas las hebras
  // y recoger los datos que han calculado copiandolo cada uno en el bufferSalida
  // salida.
  MPI_Gather( subMatriz, tamBloque * tamBloque, MPI_INT, bufferSalida, sizeof(int) * tamBloque * tamBloque, MPI_PACKED, 0, MPI_COMM_WORLD );

  // Tras esto desempaquetamos que sería la operación inversa al empaquetado.
  if (idProceso == 0){
    MPI_Type_vector(tamBloque, tamBloque, nverts, MPI_INT, &MPI_BLOQUE);
    MPI_Type_commit(&MPI_BLOQUE);

    int posicion = 0;

    for (int i = 0; i < numeroProcesos; i++) {
      filaSubmatriz = i / sqrtP;
      columnaSubmatriz = i % sqrtP;
      comienzo = columnaSubmatriz * tamBloque + filaSubmatriz * tamBloque * tamBloque * sqrtP;

      MPI_Unpack(bufferSalida, sizeof(int) * nverts * nverts, &posicion, G->getPtrMatriz() + comienzo, 1, MPI_BLOQUE,MPI_COMM_WORLD);
    }

    MPI_Type_free(&MPI_BLOQUE);
  }

  if(idProceso == 0){
    cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl;
    G->imprime();
    #if !COUT
        cout.clear();
    #endif
    cout << endl << "Tiempo gastado = "<< t << endl << endl;

    guardaEnArchivo(nverts, t);

    // Elimino el puntero de ptrInicioMatriz
    ptrInicioMatriz = NULL;
    delete ptrInicioMatriz;

    // Elimino el objeto grafo usando su destructor
    delete G;
  }

  MPI_Finalize();

  delete [] filak;
  delete [] columnak;

  return EXIT_SUCCESS;
}
