#include <iostream>
#include <cstdlib>

#include <Types.hpp>

int main (int argc, char** argv)
{
  int nelems = 10;

  if (argc > 1) {
    nelems = std::atoi(argv[1]);
  }

  if (nelems < 1) {
    std::cerr << "Invalid number of elements: " << nelems << std::endl;
    std::exit(1);
  }

  Region region(nelems);


  return 0;
}
