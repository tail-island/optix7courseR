#include <cstdlib>
#include <iostream>

#include "HelloOptix.h"

int main(int argc, char **argv) {
  try {
    osc::helloOptix();

  } catch (std::runtime_error &e) {
    std::cout << "FATAL ERROR: " << e.what() << "\n";
    std::exit(1);
  }

  return 0;
}
