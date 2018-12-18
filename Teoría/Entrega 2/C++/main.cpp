#include <iostream>
#include <vector>

#define MAXIMO 50
#define ITERS 25

using namespace std ;


std::vector<double> TransformacionIterativa(std::vector<double> entrada){

  std::vector<double> anterior = entrada, siguiente;
  double x_i, x_i_anterior, x_i_siguiente;
  double res_parcial;

  siguiente.resize(anterior.size());
  int tamanio = anterior.size();
  for (size_t k = 0; k < ITERS; k++) {
    for (size_t i = 0; i < tamanio ; i++) {
      if (i == 0) {
        x_i_anterior = anterior[tamanio-1];
        x_i = anterior[i];
        x_i_siguiente = anterior[i+1];
      }else if (i == tamanio-1) {
        x_i_anterior = anterior[i-1];
        x_i = anterior[i];
        x_i_siguiente = anterior[(i+1)%tamanio];
      }else{
        x_i_anterior = anterior[i-1];
        x_i = anterior[i];
        x_i_siguiente = anterior[i+1];
      }
      std::cout << " k = " << k <<                  endl;
      std::cout << "  - x_i_-1 = " << x_i_anterior  << endl;
      std::cout << "  - x_i_   = " << x_i           << endl;
      std::cout << "  - x_i_+1 = " << x_i_siguiente << endl;

      res_parcial = (x_i_anterior - x_i + x_i_siguiente) / 2;

      std::cout << "Nuevo valor: " << res_parcial << endl;
      siguiente[i] = res_parcial;
    }
    std::cout << endl;
    std::cout << "Tras la iter: " << k << " nuevo vector: ";
    for (size_t i = 0; i < MAXIMO; i++)
      std::cout << "[" << siguiente[i] << "] ";
    std::cout << endl << endl;
  }

  return siguiente;
}

int main(int argc, char const *argv[]) {

   std::vector<double> entrada, salida;


   // Relleno el vector y demás
   salida.resize(MAXIMO);
   entrada.resize(MAXIMO);
   for (size_t i = 0; i < entrada.size(); i++)
     //entrada[i] = rand() % MAXIMO;
     entrada[i] = i;

    std::cout << "Entrada = ";
    for (size_t i = 0; i < MAXIMO; i++)
      std::cout << "[" << entrada[i] << "] ";
    std::cout << endl;

    salida = TransformacionIterativa(entrada);

    std::cout << "Salida = ";
    for (size_t i = 0; i < MAXIMO; i++)
      std::cout << "[" << salida[i] << "] ";
    std::cout << endl;

  return 0;
}
