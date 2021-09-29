#include <cstdlib>
#include <iostream>

#include "HelloOptix.h"

int main(int Argc, char **Argv) {
  try {
    osc::helloOptiX();

  } catch (std::runtime_error &E) {
    std::cout << "FATAL ERROR: " << E.what() << "\n";
    std::exit(1);
  }

  return 0;
}
