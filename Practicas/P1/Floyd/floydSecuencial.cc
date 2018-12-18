#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <time.h>
#include "Graph.h"

using namespace std;

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

int main (int argc, char **argv){

	if (argc != 2){
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}

	Graph G;
	G.lee(argv[1]);
  cout << "Grafo:" << endl;
  G.imprime();

	int nverts = G.vertices;

  // Algoritmo Floyd SECUENCIAL a)
	double t1=clock();
    for(int k = 0; k < nverts; k++){
      for(int i = 0; i < nverts; i++){
        for(int j = 0; j < nverts; j++){
          if (i != j && i != k && j != k) {
            int vikj = min(G.arista(i,k) + G.arista(k,j), G.arista(i,j));
            G.inserta_arista(i,j,vikj);
          }
        }
      }
    }
	double t2 = clock();
  double tSecuencial = (t2-t1) / CLOCKS_PER_SEC;

  cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl << endl;
  G.imprime();
  cout << endl;

  cout<< " -> Tiempo Secuencial: " << tSecuencial << endl;
  G.obtenMasLargo();
  G.obtenMasCorto();
  cout << endl;

	string archivo = "output/floydSecuencial.dat";
	guardarArchivo(archivo, nverts, tSecuencial);
}
