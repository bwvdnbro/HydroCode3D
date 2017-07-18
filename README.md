# HydroCode3D

3D hydro code that uses a fixed grid. The code is written in (bad) C++ to run a
3D hydrostatic test for comparison with a Lagrangian mesh-free method.

To compile, use
```
g++ -O3 -fopenmp -ffast-math -Wall -Werror -o hydro_3d main.cpp
```
