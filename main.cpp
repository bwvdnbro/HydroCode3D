/*******************************************************************************
 * This file is part of HydroCode3D
 * Copyright (C) 2017 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 *
 * HydroCode3D is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HydroCode3D is distributed in the hope that it will be useful,
 * but WITOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with HydroCode3D. If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

#include "RiemannSolver.hpp"
#include "Timer.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <vector>

//#define XMIN (-200.)
//#define XMAX 200.
//#define YMIN (-200.)
//#define YMAX 200.
//#define ZMIN (-200.)
//#define ZMAX 200.
#define XMIN 0.
#define XMAX 1.
#define YMIN 0.
#define YMAX 1.
#define ZMIN 0.
#define ZMAX 1.

#define NCELL_X 100
#define NCELL_Y 10
#define NCELL_Z 10
#define GAMMA (5. / 3.)
#define DT 0.001
#define NSTEP 100
#define SNAPSTEP 10

#define BETA 0.5

#define BOUNDARIES_PERIODIC 0
#define BOUNDARIES_OPEN 1
#define BOUNDARIES_REFLECTIVE 2
#define BOUNDARIES_DISC_PATCH 3
#define BOUNDARIES_DISC_PATCH_SUP 4

#define BOUNDARIES BOUNDARIES_PERIODIC

#define EOS_IDEAL 0
#define EOS_ISOTHERMAL 1

#define EOS EOS_ISOTHERMAL

#define ISOTHERMAL_U 20.26785

#define POTENTIAL_NONE 0
#define POTENTIAL_DISC 1
#define POTENTIAL_DISC_SUP 2

#define POTENTIAL POTENTIAL_NONE

#define G 4.30097e-03

#define DISC_SUP_GROWTH_TIME 1000.

/// end of customizable parameters

#define BOXSIZE_X (XMAX - XMIN)
#define BOXSIZE_Y (YMAX - YMIN)
#define BOXSIZE_Z (ZMAX - ZMIN)
#define ISOTHERMAL_C (ISOTHERMAL_U * (GAMMA - 1.))
#define CELLSIZE_X (BOXSIZE_X / NCELL_X)
#define CELLSIZE_Y (BOXSIZE_Y / NCELL_Y)
#define CELLSIZE_Z (BOXSIZE_Z / NCELL_Z)
#define CELLSIZE_X_INV (1. / CELLSIZE_X)
#define CELLSIZE_Y_INV (1. / CELLSIZE_Y)
#define CELLSIZE_Z_INV (1. / CELLSIZE_Z)
#define CELL_VOLUME (CELLSIZE_X * CELLSIZE_Y * CELLSIZE_Z)
#define CELL_VOLUME_INV (1. / CELL_VOLUME)
#define CELL_AREA_X (CELLSIZE_Y * CELLSIZE_Z)
#define CELL_AREA_Y (CELLSIZE_X * CELLSIZE_Z)
#define CELL_AREA_Z (CELLSIZE_X * CELLSIZE_Y)
#define DT_CELL_AREA_X (DT * CELL_AREA_X)
#define DT_CELL_AREA_Y (DT * CELL_AREA_Y)
#define DT_CELL_AREA_Z (DT * CELL_AREA_Z)

class Cell {
public:
  double _rho;
  double _u[3];
  double _P;

  double _grad_rho[3];
  double _grad_u[3][3];
  double _grad_P[3];

  double _m;
  double _p[3];
  double _E;

  double _midpoint[3];

  double _a[3];

  omp_lock_t _lock;

  Cell() { omp_init_lock(&_lock); }

  ~Cell() { omp_destroy_lock(&_lock); }

  void lock() { omp_set_lock(&_lock); }

  void unlock() { omp_unset_lock(&_lock); }
};

void init(Cell &cell) {
  //  cell._rho = 9.64028/400.;
  //  cell._rho = 1.24916e-02;
  if (cell._midpoint[0] < 0.5) {
    cell._rho = 1.;
    cell._P = 1.;
  } else {
    cell._rho = 0.125;
    cell._P = 0.1;
  }
  cell._u[0] = 0.;
  cell._u[1] = 0.;
  cell._u[2] = 0.;
  cell._a[0] = 0.;
  cell._a[1] = 0.;
  cell._a[2] = 0.;
}

int main(int argc, char **argv) {
#if EOS == EOS_ISOTHERMAL
  std::cout << "ISOTHERMAL_C: " << ISOTHERMAL_C << std::endl;
#endif

  int num_threads = 0;
#pragma omp parallel
  {
#pragma omp single
    { num_threads = omp_get_num_threads(); }
  }

  std::cout << "Using " << num_threads << " threads." << std::endl;

  Timer timer;
  timer.start();

  std::vector< std::vector< std::vector< Cell > > > cells(
      NCELL_X + 2, std::vector< std::vector< Cell > >(
                       NCELL_Y + 2, std::vector< Cell >(NCELL_Z + 2)));

#pragma omp parallel for default(shared)
  for (unsigned int ix = 0; ix < NCELL_X + 2; ++ix) {
    for (unsigned int iy = 0; iy < NCELL_Y + 2; ++iy) {
      for (unsigned int iz = 0; iz < NCELL_Z + 2; ++iz) {
        Cell &cell = cells[ix][iy][iz];
        cell._midpoint[0] = XMIN + (ix - 0.5) * CELLSIZE_X;
        cell._midpoint[1] = YMIN + (iy - 0.5) * CELLSIZE_Y;
        cell._midpoint[2] = ZMIN + (iz - 0.5) * CELLSIZE_Z;
      }
    }
  }

#pragma omp parallel for default(shared)
  for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
    for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        Cell &cell = cells[ix][iy][iz];
        init(cell);

#if EOS == EOS_ISOTHERMAL
        cell._P = ISOTHERMAL_C * cell._rho;
#endif

        cell._m = cell._rho * CELL_VOLUME;
        cell._p[0] = cell._m * cell._u[0];
        cell._p[1] = cell._m * cell._u[1];
        cell._p[2] = cell._m * cell._u[2];
        cell._E = cell._P * CELL_VOLUME / (GAMMA - 1.) +
                  0.5 * (cell._u[0] * cell._p[0] + cell._u[1] * cell._p[1] +
                         cell._u[2] * cell._p[2]);
      }
    }
  }

  RiemannSolver solver(GAMMA);

  for (unsigned int istep = 0; istep < NSTEP; ++istep) {

#if POTENTIAL == POTENTIAL_DISC
    for (unsigned int i = 0; i < NCELL_TOTAL; ++i) {
      double reduction_factor = 1.;
      if (istep * DT < 5. * 48.) {
        reduction_factor = istep * DT / 5. / 48.;
      }
      double a = reduction_factor * 2. * M_PI * 10. *
                 std::tanh(cells[i]._midpoint[0] * 0.01);
      a *= G;
      cells[i]._a[0] = a;
      cells[i]._p[0] -= 0.5 * DT * a * cells[i]._m;
    }
#elif POTENTIAL == POTENTIAL_DISC_SUP
    for (unsigned int i = 0; i < NCELL_TOTAL; ++i) {
      double reduction_factor = 1.;
      if (istep * DT < DISC_SUP_GROWTH_TIME) {
        reduction_factor = istep * DT / DISC_SUP_GROWTH_TIME;
      }
      double a;
      if (cells[i]._midpoint[0] > -300. && cells[i]._midpoint[0] < 300.) {
        a = 2. * M_PI * 10. * std::tanh(cells[i]._midpoint[0] * 0.01);
      } else if (cells[i]._midpoint[0] > -350. &&
                 cells[i]._midpoint[0] < 350.) {
        a = 2. * M_PI * 10. * std::tanh(cells[i]._midpoint[0] * 0.01) *
            (0.5 +
             0.5 * std::cos(M_PI * (std::abs(cells[i]._midpoint[0]) - 300.) *
                            0.02));
      } else {
        a = 0.;
      }
      a *= reduction_factor * G;
      cells[i]._a[0] = a;
      cells[i]._p[0] -= 0.5 * DT * a * cells[i]._m;
    }
#endif

#pragma omp parallel for default(shared)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
          Cell &cell = cells[ix][iy][iz];
          cell._rho = cell._m * CELL_VOLUME_INV;
          const double minv = 1. / cell._m;
          cell._u[0] = cell._p[0] * minv;
          cell._u[1] = cell._p[1] * minv;
          cell._u[2] = cell._p[2] * minv;
#if EOS == EOS_IDEAL
          cell._P =
              (GAMMA - 1.) * (cell._E * CELL_VOLUME_INV -
                              0.5 * cell._rho * (cell._u[0] * cell._u[0] +
                                                 cell._u[1] * cell._u[1] +
                                                 cell._u[2] * cell._u[2]));
#elif EOS == EOS_ISOTHERMAL
          cell._P = ISOTHERMAL_C * cell._rho;
#else
#error "No equation of state selected!"
#endif
        }
      }
    }

    if (istep % SNAPSTEP == 0) {
      std::cout << "step " << istep << " of " << NSTEP << std::endl;
      std::stringstream filename;
      filename << "snap_";
      filename.fill('0');
      filename.width(3);
      filename << (istep / SNAPSTEP);
      filename << ".txt";
      std::ofstream ofile(filename.str().c_str());
      for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
        for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
          for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
            Cell &cell = cells[ix][iy][iz];
            ofile << cell._midpoint[0] << "\t" << cell._midpoint[1] << "\t"
                  << cell._midpoint[2] << "\t" << cell._rho << "\t"
                  << cell._u[0] << "\t" << cell._u[1] << "\t" << cell._u[2]
                  << "\t" << cell._P << "\t" << cell._a[0] << "\t" << cell._a[1]
                  << "\t" << cell._a[2] << "\n";
          }
        }
      }
      ofile.close();
    }

#if BOUNDARIES == BOUNDARIES_PERIODIC
#pragma omp parallel for default(shared)
    for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        cells[0][iy][iz]._rho = cells[NCELL_X][iy][iz]._rho;
        cells[0][iy][iz]._u[0] = cells[NCELL_X][iy][iz]._u[0];
        cells[0][iy][iz]._u[1] = cells[NCELL_X][iy][iz]._u[1];
        cells[0][iy][iz]._u[2] = cells[NCELL_X][iy][iz]._u[2];
        cells[0][iy][iz]._P = cells[NCELL_X][iy][iz]._P;
        cells[NCELL_X + 1][iy][iz]._rho = cells[1][iy][iz]._rho;
        cells[NCELL_X + 1][iy][iz]._u[0] = cells[1][iy][iz]._u[0];
        cells[NCELL_X + 1][iy][iz]._u[1] = cells[1][iy][iz]._u[1];
        cells[NCELL_X + 1][iy][iz]._u[2] = cells[1][iy][iz]._u[2];
        cells[NCELL_X + 1][iy][iz]._P = cells[1][iy][iz]._P;
      }
    }
#pragma omp parallel for default(shared)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        cells[ix][0][iz]._rho = cells[ix][NCELL_Y][iz]._rho;
        cells[ix][0][iz]._u[0] = cells[ix][NCELL_Y][iz]._u[0];
        cells[ix][0][iz]._u[1] = cells[ix][NCELL_Y][iz]._u[1];
        cells[ix][0][iz]._u[2] = cells[ix][NCELL_Y][iz]._u[2];
        cells[ix][0][iz]._P = cells[ix][NCELL_Y][iz]._P;
        cells[ix][NCELL_Y + 1][iz]._rho = cells[ix][1][iz]._rho;
        cells[ix][NCELL_Y + 1][iz]._u[0] = cells[ix][1][iz]._u[0];
        cells[ix][NCELL_Y + 1][iz]._u[1] = cells[ix][1][iz]._u[1];
        cells[ix][NCELL_Y + 1][iz]._u[2] = cells[ix][1][iz]._u[2];
        cells[ix][NCELL_Y + 1][iz]._P = cells[ix][1][iz]._P;
      }
    }
#pragma omp parallel for default(shared)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        cells[ix][iy][0]._rho = cells[ix][iy][NCELL_Z]._rho;
        cells[ix][iy][0]._u[0] = cells[ix][iy][NCELL_Z]._u[0];
        cells[ix][iy][0]._u[1] = cells[ix][iy][NCELL_Z]._u[1];
        cells[ix][iy][0]._u[2] = cells[ix][iy][NCELL_Z]._u[2];
        cells[ix][iy][0]._P = cells[ix][iy][NCELL_Z]._P;
        cells[ix][iy][NCELL_Z + 1]._rho = cells[ix][iy][1]._rho;
        cells[ix][iy][NCELL_Z + 1]._u[0] = cells[ix][iy][1]._u[0];
        cells[ix][iy][NCELL_Z + 1]._u[1] = cells[ix][iy][1]._u[1];
        cells[ix][iy][NCELL_Z + 1]._u[2] = cells[ix][iy][1]._u[2];
        cells[ix][iy][NCELL_Z + 1]._P = cells[ix][iy][1]._P;
      }
    }
#elif BOUNDARIES == BOUNDARIES_OPEN
    cells[0]._rho = cells[1]._rho;
    cells[0]._u = cells[1]._u;
    cells[0]._P = cells[1]._P;
    cells[NCELL + 1]._rho = cells[NCELL]._rho;
    cells[NCELL + 1]._u = cells[NCELL]._u;
    cells[NCELL + 1]._P = cells[NCELL]._P;
#elif BOUNDARIES == BOUNDARIES_REFLECTIVE
    cells[0]._rho = cells[1]._rho;
    cells[0]._u = -cells[1]._u;
    cells[0]._P = cells[1]._P;
    cells[NCELL + 1]._rho = cells[NCELL]._rho;
    cells[NCELL + 1]._u = -cells[NCELL]._u;
    cells[NCELL + 1]._P = cells[NCELL]._P;
#elif BOUNDARIES == BOUNDARIES_DISC_PATCH
    const double coshx = std::cosh(cells[0]._midpoint * 0.01);
    const double coshx2 = coshx * coshx;
    cells[0]._rho = 0.05 / coshx2;
    cells[0]._u = 0.;
    cells[0]._P = ISOTHERMAL_C * cells[0]._rho;
    // symmetry
    cells[NCELL + 1]._rho = cells[0]._rho;
    cells[NCELL + 1]._u = 0.;
    cells[NCELL + 1]._P = cells[0]._P;
#elif BOUNDARIES == BOUNDARIES_DISC_PATCH_SUP
    cells[0]._rho = cells[1]._rho;
    cells[0]._u = 0.;
    cells[0]._P = cells[1]._P;
    cells[NCELL + 1]._rho = cells[NCELL]._rho;
    cells[NCELL + 1]._u = 0.;
    cells[NCELL + 1]._P = cells[NCELL]._P;
#else
#error "No boundary conditions chosen!"
#endif

#pragma omp parallel for default(shared)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
          Cell &cell = cells[ix][iy][iz];

          cell._grad_rho[0] =
              0.5 * CELLSIZE_X_INV *
              (cells[ix + 1][iy][iz]._rho - cells[ix - 1][iy][iz]._rho);
          cell._grad_rho[1] =
              0.5 * CELLSIZE_Y_INV *
              (cells[ix][iy + 1][iz]._rho - cells[ix][iy - 1][iz]._rho);
          cell._grad_rho[2] =
              0.5 * CELLSIZE_Z_INV *
              (cells[ix][iy][iz + 1]._rho - cells[ix][iy][iz - 1]._rho);

          cell._grad_u[0][0] =
              0.5 * CELLSIZE_X_INV *
              (cells[ix + 1][iy][iz]._u[0] - cells[ix - 1][iy][iz]._u[0]);
          cell._grad_u[0][1] =
              0.5 * CELLSIZE_Y_INV *
              (cells[ix][iy + 1][iz]._u[0] - cells[ix][iy - 1][iz]._u[0]);
          cell._grad_u[0][2] =
              0.5 * CELLSIZE_Z_INV *
              (cells[ix][iy][iz + 1]._u[0] - cells[ix][iy][iz - 1]._u[0]);
          cell._grad_u[1][0] =
              0.5 * CELLSIZE_X_INV *
              (cells[ix + 1][iy][iz]._u[1] - cells[ix - 1][iy][iz]._u[1]);
          cell._grad_u[1][1] =
              0.5 * CELLSIZE_Y_INV *
              (cells[ix][iy + 1][iz]._u[1] - cells[ix][iy - 1][iz]._u[1]);
          cell._grad_u[1][2] =
              0.5 * CELLSIZE_Z_INV *
              (cells[ix][iy][iz + 1]._u[1] - cells[ix][iy][iz - 1]._u[1]);
          cell._grad_u[2][0] =
              0.5 * CELLSIZE_X_INV *
              (cells[ix + 1][iy][iz]._u[2] - cells[ix - 1][iy][iz]._u[2]);
          cell._grad_u[2][1] =
              0.5 * CELLSIZE_Y_INV *
              (cells[ix][iy + 1][iz]._u[2] - cells[ix][iy - 1][iz]._u[2]);
          cell._grad_u[2][2] =
              0.5 * CELLSIZE_Z_INV *
              (cells[ix][iy][iz + 1]._u[2] - cells[ix][iy][iz - 1]._u[2]);

          cell._grad_P[0] = 0.5 * CELLSIZE_X_INV * (cells[ix + 1][iy][iz]._P -
                                                    cells[ix - 1][iy][iz]._P);
          cell._grad_P[1] = 0.5 * CELLSIZE_Y_INV * (cells[ix][iy + 1][iz]._P -
                                                    cells[ix][iy - 1][iz]._P);
          cell._grad_P[2] = 0.5 * CELLSIZE_Z_INV * (cells[ix][iy][iz + 1]._P -
                                                    cells[ix][iy][iz - 1]._P);

          double alpha, phi_ngb_max, phi_ext_max, phi_ngb_min, phi_ext_min;
          {
            // density
            phi_ngb_max = std::max(cells[ix - 1][iy][iz]._rho,
                                   cells[ix + 1][iy][iz]._rho);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy - 1][iz]._rho);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy + 1][iz]._rho);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz - 1]._rho);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz + 1]._rho);
            phi_ngb_max -= cell._rho;
            phi_ngb_min = std::min(cells[ix - 1][iy][iz]._rho,
                                   cells[ix + 1][iy][iz]._rho);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy - 1][iz]._rho);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy + 1][iz]._rho);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz - 1]._rho);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz + 1]._rho);
            phi_ngb_min -= cell._rho;
            phi_ext_max = 0.5 * CELLSIZE_X *
                          std::max(cell._grad_rho[0], -cell._grad_rho[0]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Y * cell._grad_rho[1]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Y * cell._grad_rho[1]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Z * cell._grad_rho[2]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Z * cell._grad_rho[2]);
            phi_ext_min = 0.5 * CELLSIZE_X *
                          std::min(cell._grad_rho[0], -cell._grad_rho[0]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Y * cell._grad_rho[1]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Y * cell._grad_rho[1]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Z * cell._grad_rho[2]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Z * cell._grad_rho[2]);

            alpha = std::min(1., BETA * std::min(phi_ngb_max / phi_ext_max,
                                                 phi_ngb_min / phi_ext_min));
            cell._grad_rho[0] *= alpha;
            cell._grad_rho[1] *= alpha;
            cell._grad_rho[2] *= alpha;
          }
          {
            // velocity x
            phi_ngb_max = std::max(cells[ix - 1][iy][iz]._u[0],
                                   cells[ix + 1][iy][iz]._u[0]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy - 1][iz]._u[0]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy + 1][iz]._u[0]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz - 1]._u[0]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz + 1]._u[0]);
            phi_ngb_max -= cell._u[0];
            phi_ngb_min = std::min(cells[ix - 1][iy][iz]._u[0],
                                   cells[ix + 1][iy][iz]._u[0]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy - 1][iz]._u[0]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy + 1][iz]._u[0]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz - 1]._u[0]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz + 1]._u[0]);
            phi_ngb_min -= cell._u[0];
            phi_ext_max = 0.5 * CELLSIZE_X *
                          std::max(cell._grad_u[0][0], -cell._grad_u[0][0]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Y * cell._grad_u[0][1]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Y * cell._grad_u[0][1]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Z * cell._grad_u[0][2]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Z * cell._grad_u[0][2]);
            phi_ext_min = 0.5 * CELLSIZE_X *
                          std::min(cell._grad_u[0][0], -cell._grad_u[0][0]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Y * cell._grad_u[0][1]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Y * cell._grad_u[0][1]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Z * cell._grad_u[0][2]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Z * cell._grad_u[0][2]);

            alpha = std::min(1., BETA * std::min(phi_ngb_max / phi_ext_max,
                                                 phi_ngb_min / phi_ext_min));
            cell._grad_u[0][0] *= alpha;
            cell._grad_u[0][1] *= alpha;
            cell._grad_u[0][2] *= alpha;
          }
          {
            // velocity y
            phi_ngb_max = std::max(cells[ix - 1][iy][iz]._u[1],
                                   cells[ix + 1][iy][iz]._u[1]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy - 1][iz]._u[1]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy + 1][iz]._u[1]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz - 1]._u[1]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz + 1]._u[1]);
            phi_ngb_max -= cell._u[1];
            phi_ngb_min = std::min(cells[ix - 1][iy][iz]._u[1],
                                   cells[ix + 1][iy][iz]._u[1]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy - 1][iz]._u[1]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy + 1][iz]._u[1]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz - 1]._u[1]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz + 1]._u[1]);
            phi_ngb_min -= cell._u[1];
            phi_ext_max = 0.5 * CELLSIZE_X *
                          std::max(cell._grad_u[1][0], -cell._grad_u[1][0]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Y * cell._grad_u[1][1]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Y * cell._grad_u[1][1]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Z * cell._grad_u[1][2]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Z * cell._grad_u[1][2]);
            phi_ext_min = 0.5 * CELLSIZE_X *
                          std::min(cell._grad_u[1][0], -cell._grad_u[1][0]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Y * cell._grad_u[1][1]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Y * cell._grad_u[1][1]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Z * cell._grad_u[1][2]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Z * cell._grad_u[1][2]);

            alpha = std::min(1., BETA * std::min(phi_ngb_max / phi_ext_max,
                                                 phi_ngb_min / phi_ext_min));
            cell._grad_u[1][0] *= alpha;
            cell._grad_u[1][1] *= alpha;
            cell._grad_u[1][2] *= alpha;
          }
          {
            // velocity z
            phi_ngb_max = std::max(cells[ix - 1][iy][iz]._u[2],
                                   cells[ix + 1][iy][iz]._u[2]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy - 1][iz]._u[2]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy + 1][iz]._u[2]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz - 1]._u[2]);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz + 1]._u[2]);
            phi_ngb_max -= cell._u[2];
            phi_ngb_min = std::min(cells[ix - 1][iy][iz]._u[2],
                                   cells[ix + 1][iy][iz]._u[2]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy - 1][iz]._u[2]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy + 1][iz]._u[2]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz - 1]._u[2]);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz + 1]._u[2]);
            phi_ngb_min -= cell._u[2];
            phi_ext_max = 0.5 * CELLSIZE_X *
                          std::max(cell._grad_u[2][0], -cell._grad_u[2][0]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Y * cell._grad_u[2][1]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Y * cell._grad_u[2][1]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Z * cell._grad_u[2][2]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Z * cell._grad_u[2][2]);
            phi_ext_min = 0.5 * CELLSIZE_X *
                          std::min(cell._grad_u[2][0], -cell._grad_u[2][0]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Y * cell._grad_u[2][1]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Y * cell._grad_u[2][1]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Z * cell._grad_u[2][2]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Z * cell._grad_u[2][2]);

            alpha = std::min(1., BETA * std::min(phi_ngb_max / phi_ext_max,
                                                 phi_ngb_min / phi_ext_min));
            cell._grad_u[2][0] *= alpha;
            cell._grad_u[2][1] *= alpha;
            cell._grad_u[2][2] *= alpha;
          }
          {
            // pressure
            phi_ngb_max =
                std::max(cells[ix - 1][iy][iz]._P, cells[ix + 1][iy][iz]._P);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy - 1][iz]._P);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy + 1][iz]._P);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz - 1]._P);
            phi_ngb_max = std::max(phi_ngb_max, cells[ix][iy][iz + 1]._P);
            phi_ngb_max -= cell._P;
            phi_ngb_min =
                std::min(cells[ix - 1][iy][iz]._P, cells[ix + 1][iy][iz]._P);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy - 1][iz]._P);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy + 1][iz]._P);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz - 1]._P);
            phi_ngb_min = std::min(phi_ngb_min, cells[ix][iy][iz + 1]._P);
            phi_ngb_min -= cell._P;
            phi_ext_max =
                0.5 * CELLSIZE_X * std::max(cell._grad_P[0], -cell._grad_P[0]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Y * cell._grad_P[1]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Y * cell._grad_P[1]);
            phi_ext_max =
                std::max(phi_ext_max, 0.5 * CELLSIZE_Z * cell._grad_P[2]);
            phi_ext_max =
                std::max(phi_ext_max, -0.5 * CELLSIZE_Z * cell._grad_P[2]);
            phi_ext_min =
                0.5 * CELLSIZE_X * std::min(cell._grad_P[0], -cell._grad_P[0]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Y * cell._grad_P[1]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Y * cell._grad_P[1]);
            phi_ext_min =
                std::min(phi_ext_min, 0.5 * CELLSIZE_Z * cell._grad_P[2]);
            phi_ext_min =
                std::min(phi_ext_min, -0.5 * CELLSIZE_Z * cell._grad_P[2]);

            alpha = std::min(1., BETA * std::min(phi_ngb_max / phi_ext_max,
                                                 phi_ngb_min / phi_ext_min));
            cell._grad_P[0] *= alpha;
            cell._grad_P[1] *= alpha;
            cell._grad_P[2] *= alpha;
          }
        }
      }
    }

#if BOUNDARIES == BOUNDARIES_PERIODIC
#pragma omp parallel for default(shared)
    for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        cells[0][iy][iz]._grad_rho[0] = cells[NCELL_X][iy][iz]._grad_rho[0];
        cells[0][iy][iz]._grad_rho[1] = cells[NCELL_X][iy][iz]._grad_rho[1];
        cells[0][iy][iz]._grad_rho[2] = cells[NCELL_X][iy][iz]._grad_rho[2];
        cells[0][iy][iz]._grad_u[0][0] = cells[NCELL_X][iy][iz]._grad_u[0][0];
        cells[0][iy][iz]._grad_u[0][1] = cells[NCELL_X][iy][iz]._grad_u[0][1];
        cells[0][iy][iz]._grad_u[0][2] = cells[NCELL_X][iy][iz]._grad_u[0][2];
        cells[0][iy][iz]._grad_u[1][0] = cells[NCELL_X][iy][iz]._grad_u[1][0];
        cells[0][iy][iz]._grad_u[1][1] = cells[NCELL_X][iy][iz]._grad_u[1][1];
        cells[0][iy][iz]._grad_u[1][2] = cells[NCELL_X][iy][iz]._grad_u[1][2];
        cells[0][iy][iz]._grad_u[2][0] = cells[NCELL_X][iy][iz]._grad_u[2][0];
        cells[0][iy][iz]._grad_u[2][1] = cells[NCELL_X][iy][iz]._grad_u[2][1];
        cells[0][iy][iz]._grad_u[2][2] = cells[NCELL_X][iy][iz]._grad_u[2][2];
        cells[0][iy][iz]._grad_P[0] = cells[NCELL_X][iy][iz]._grad_P[0];
        cells[0][iy][iz]._grad_P[1] = cells[NCELL_X][iy][iz]._grad_P[1];
        cells[0][iy][iz]._grad_P[2] = cells[NCELL_X][iy][iz]._grad_P[2];
        cells[NCELL_X + 1][iy][iz]._grad_rho[0] = cells[1][iy][iz]._grad_rho[0];
        cells[NCELL_X + 1][iy][iz]._grad_rho[1] = cells[1][iy][iz]._grad_rho[1];
        cells[NCELL_X + 1][iy][iz]._grad_rho[2] = cells[1][iy][iz]._grad_rho[2];
        cells[NCELL_X + 1][iy][iz]._grad_u[0][0] =
            cells[1][iy][iz]._grad_u[0][0];
        cells[NCELL_X + 1][iy][iz]._grad_u[0][1] =
            cells[1][iy][iz]._grad_u[0][1];
        cells[NCELL_X + 1][iy][iz]._grad_u[0][2] =
            cells[1][iy][iz]._grad_u[0][2];
        cells[NCELL_X + 1][iy][iz]._grad_u[1][0] =
            cells[1][iy][iz]._grad_u[1][0];
        cells[NCELL_X + 1][iy][iz]._grad_u[1][1] =
            cells[1][iy][iz]._grad_u[1][1];
        cells[NCELL_X + 1][iy][iz]._grad_u[1][2] =
            cells[1][iy][iz]._grad_u[1][2];
        cells[NCELL_X + 1][iy][iz]._grad_u[2][0] =
            cells[1][iy][iz]._grad_u[2][0];
        cells[NCELL_X + 1][iy][iz]._grad_u[2][1] =
            cells[1][iy][iz]._grad_u[2][1];
        cells[NCELL_X + 1][iy][iz]._grad_u[2][2] =
            cells[1][iy][iz]._grad_u[2][2];
        cells[NCELL_X + 1][iy][iz]._grad_P[0] = cells[1][iy][iz]._grad_P[0];
        cells[NCELL_X + 1][iy][iz]._grad_P[1] = cells[1][iy][iz]._grad_P[1];
        cells[NCELL_X + 1][iy][iz]._grad_P[2] = cells[1][iy][iz]._grad_P[2];
      }
    }
#pragma omp parallel for default(shared)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        cells[ix][0][iz]._grad_rho[0] = cells[ix][NCELL_Y][iz]._grad_rho[0];
        cells[ix][0][iz]._grad_rho[1] = cells[ix][NCELL_Y][iz]._grad_rho[1];
        cells[ix][0][iz]._grad_rho[2] = cells[ix][NCELL_Y][iz]._grad_rho[2];
        cells[ix][0][iz]._grad_u[0][0] = cells[ix][NCELL_Y][iz]._grad_u[0][0];
        cells[ix][0][iz]._grad_u[0][1] = cells[ix][NCELL_Y][iz]._grad_u[0][1];
        cells[ix][0][iz]._grad_u[0][2] = cells[ix][NCELL_Y][iz]._grad_u[0][2];
        cells[ix][0][iz]._grad_u[1][0] = cells[ix][NCELL_Y][iz]._grad_u[1][0];
        cells[ix][0][iz]._grad_u[1][1] = cells[ix][NCELL_Y][iz]._grad_u[1][1];
        cells[ix][0][iz]._grad_u[1][2] = cells[ix][NCELL_Y][iz]._grad_u[1][2];
        cells[ix][0][iz]._grad_u[2][0] = cells[ix][NCELL_Y][iz]._grad_u[2][0];
        cells[ix][0][iz]._grad_u[2][1] = cells[ix][NCELL_Y][iz]._grad_u[2][1];
        cells[ix][0][iz]._grad_u[2][2] = cells[ix][NCELL_Y][iz]._grad_u[2][2];
        cells[ix][0][iz]._grad_P[0] = cells[ix][NCELL_Y][iz]._grad_P[0];
        cells[ix][0][iz]._grad_P[1] = cells[ix][NCELL_Y][iz]._grad_P[1];
        cells[ix][0][iz]._grad_P[2] = cells[ix][NCELL_Y][iz]._grad_P[2];
        cells[ix][NCELL_Y + 1][iz]._grad_rho[0] = cells[ix][1][iz]._grad_rho[0];
        cells[ix][NCELL_Y + 1][iz]._grad_rho[1] = cells[ix][1][iz]._grad_rho[1];
        cells[ix][NCELL_Y + 1][iz]._grad_rho[2] = cells[ix][1][iz]._grad_rho[2];
        cells[ix][NCELL_Y + 1][iz]._grad_u[0][0] =
            cells[ix][1][iz]._grad_u[0][0];
        cells[ix][NCELL_Y + 1][iz]._grad_u[0][1] =
            cells[ix][1][iz]._grad_u[0][1];
        cells[ix][NCELL_Y + 1][iz]._grad_u[0][2] =
            cells[ix][1][iz]._grad_u[0][2];
        cells[ix][NCELL_Y + 1][iz]._grad_u[1][0] =
            cells[ix][1][iz]._grad_u[1][0];
        cells[ix][NCELL_Y + 1][iz]._grad_u[1][1] =
            cells[ix][1][iz]._grad_u[1][1];
        cells[ix][NCELL_Y + 1][iz]._grad_u[1][2] =
            cells[ix][1][iz]._grad_u[1][2];
        cells[ix][NCELL_Y + 1][iz]._grad_u[2][0] =
            cells[ix][1][iz]._grad_u[2][0];
        cells[ix][NCELL_Y + 1][iz]._grad_u[2][1] =
            cells[ix][1][iz]._grad_u[2][1];
        cells[ix][NCELL_Y + 1][iz]._grad_u[2][2] =
            cells[ix][1][iz]._grad_u[2][2];
        cells[ix][NCELL_Y + 1][iz]._grad_P[0] = cells[ix][1][iz]._grad_P[0];
        cells[ix][NCELL_Y + 1][iz]._grad_P[1] = cells[ix][1][iz]._grad_P[1];
        cells[ix][NCELL_Y + 1][iz]._grad_P[2] = cells[ix][1][iz]._grad_P[2];
      }
    }
#pragma omp parallel for default(shared)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        cells[ix][iy][0]._grad_rho[0] = cells[ix][iy][NCELL_Z]._grad_rho[0];
        cells[ix][iy][0]._grad_rho[1] = cells[ix][iy][NCELL_Z]._grad_rho[1];
        cells[ix][iy][0]._grad_rho[2] = cells[ix][iy][NCELL_Z]._grad_rho[2];
        cells[ix][iy][0]._grad_u[0][0] = cells[ix][iy][NCELL_Z]._grad_u[0][0];
        cells[ix][iy][0]._grad_u[0][1] = cells[ix][iy][NCELL_Z]._grad_u[0][1];
        cells[ix][iy][0]._grad_u[0][2] = cells[ix][iy][NCELL_Z]._grad_u[0][2];
        cells[ix][iy][0]._grad_u[1][0] = cells[ix][iy][NCELL_Z]._grad_u[1][0];
        cells[ix][iy][0]._grad_u[1][1] = cells[ix][iy][NCELL_Z]._grad_u[1][1];
        cells[ix][iy][0]._grad_u[1][2] = cells[ix][iy][NCELL_Z]._grad_u[1][2];
        cells[ix][iy][0]._grad_u[2][0] = cells[ix][iy][NCELL_Z]._grad_u[2][0];
        cells[ix][iy][0]._grad_u[2][1] = cells[ix][iy][NCELL_Z]._grad_u[2][1];
        cells[ix][iy][0]._grad_u[2][2] = cells[ix][iy][NCELL_Z]._grad_u[2][2];
        cells[ix][iy][0]._grad_P[0] = cells[ix][iy][NCELL_Z]._grad_P[0];
        cells[ix][iy][0]._grad_P[1] = cells[ix][iy][NCELL_Z]._grad_P[1];
        cells[ix][iy][0]._grad_P[2] = cells[ix][iy][NCELL_Z]._grad_P[2];
        cells[ix][iy][NCELL_Z + 1]._grad_rho[0] = cells[ix][iy][1]._grad_rho[0];
        cells[ix][iy][NCELL_Z + 1]._grad_rho[1] = cells[ix][iy][1]._grad_rho[1];
        cells[ix][iy][NCELL_Z + 1]._grad_rho[2] = cells[ix][iy][1]._grad_rho[2];
        cells[ix][iy][NCELL_Z + 1]._grad_u[0][0] =
            cells[ix][iy][1]._grad_u[0][0];
        cells[ix][iy][NCELL_Z + 1]._grad_u[0][1] =
            cells[ix][iy][1]._grad_u[0][1];
        cells[ix][iy][NCELL_Z + 1]._grad_u[0][2] =
            cells[ix][iy][1]._grad_u[0][2];
        cells[ix][iy][NCELL_Z + 1]._grad_u[1][0] =
            cells[ix][iy][1]._grad_u[1][0];
        cells[ix][iy][NCELL_Z + 1]._grad_u[1][1] =
            cells[ix][iy][1]._grad_u[1][1];
        cells[ix][iy][NCELL_Z + 1]._grad_u[1][2] =
            cells[ix][iy][1]._grad_u[1][2];
        cells[ix][iy][NCELL_Z + 1]._grad_u[2][0] =
            cells[ix][iy][1]._grad_u[2][0];
        cells[ix][iy][NCELL_Z + 1]._grad_u[2][1] =
            cells[ix][iy][1]._grad_u[2][1];
        cells[ix][iy][NCELL_Z + 1]._grad_u[2][2] =
            cells[ix][iy][1]._grad_u[2][2];
        cells[ix][iy][NCELL_Z + 1]._grad_P[0] = cells[ix][iy][1]._grad_P[0];
        cells[ix][iy][NCELL_Z + 1]._grad_P[1] = cells[ix][iy][1]._grad_P[1];
        cells[ix][iy][NCELL_Z + 1]._grad_P[2] = cells[ix][iy][1]._grad_P[2];
      }
    }
#elif BOUNDARIES == BOUNDARIES_OPEN
    cells[0]._rho = cells[1]._rho;
    cells[0]._u = cells[1]._u;
    cells[0]._P = cells[1]._P;
    cells[NCELL + 1]._rho = cells[NCELL]._rho;
    cells[NCELL + 1]._u = cells[NCELL]._u;
    cells[NCELL + 1]._P = cells[NCELL]._P;
#elif BOUNDARIES == BOUNDARIES_REFLECTIVE
    cells[0]._rho = cells[1]._rho;
    cells[0]._u = -cells[1]._u;
    cells[0]._P = cells[1]._P;
    cells[NCELL + 1]._rho = cells[NCELL]._rho;
    cells[NCELL + 1]._u = -cells[NCELL]._u;
    cells[NCELL + 1]._P = cells[NCELL]._P;
#elif BOUNDARIES == BOUNDARIES_DISC_PATCH
    const double coshx = std::cosh(cells[0]._midpoint * 0.01);
    const double coshx2 = coshx * coshx;
    cells[0]._rho = 0.05 / coshx2;
    cells[0]._u = 0.;
    cells[0]._P = ISOTHERMAL_C * cells[0]._rho;
    // symmetry
    cells[NCELL + 1]._rho = cells[0]._rho;
    cells[NCELL + 1]._u = 0.;
    cells[NCELL + 1]._P = cells[0]._P;
#elif BOUNDARIES == BOUNDARIES_DISC_PATCH_SUP
    cells[0]._rho = cells[1]._rho;
    cells[0]._u = 0.;
    cells[0]._P = cells[1]._P;
    cells[NCELL + 1]._rho = cells[NCELL]._rho;
    cells[NCELL + 1]._u = 0.;
    cells[NCELL + 1]._P = cells[NCELL]._P;
#else
#error "No boundary conditions chosen!"
#endif

#pragma omp parallel for default(shared)
    for (unsigned int ix = 0; ix < NCELL_X + 2; ++ix) {
      for (unsigned int iy = 0; iy < NCELL_Y + 2; ++iy) {
        for (unsigned int iz = 0; iz < NCELL_Z + 2; ++iz) {
          Cell &cell = cells[ix][iy][iz];

          double rho, u[3], P, rho_inv;
          double grho[3], gP[3], div_u;

          rho = cell._rho;
          u[0] = cell._u[0];
          u[1] = cell._u[1];
          u[2] = cell._u[2];
          P = cell._P;
          grho[0] = cell._grad_rho[0];
          grho[1] = cell._grad_rho[1];
          grho[2] = cell._grad_rho[2];
          gP[0] = cell._grad_P[0];
          gP[1] = cell._grad_P[1];
          gP[2] = cell._grad_P[2];

          rho_inv = 1. / rho;
          div_u = cell._grad_u[0][0] + cell._grad_u[1][1] + cell._grad_u[2][2];

          cell._rho -= 0.5 * DT * (rho * div_u + u[0] * grho[0] +
                                   u[1] * grho[1] + u[2] * grho[2]);
          cell._u[0] -= 0.5 * DT * (u[0] * div_u + rho_inv * gP[0]);
          cell._u[1] -= 0.5 * DT * (u[1] * div_u + rho_inv * gP[1]);
          cell._u[2] -= 0.5 * DT * (u[2] * div_u + rho_inv * gP[2]);
          cell._P -= 0.5 * DT * (GAMMA * P * div_u + u[0] * gP[0] +
                                 u[1] * gP[1] + u[2] * gP[2]);
        }
      }
    }

#pragma omp parallel for default(shared)
    for (unsigned int ix = 0; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 0; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 0; iz < NCELL_Z + 1; ++iz) {

          Cell &lcell = cells[ix][iy][iz];
          double rhoL, uL[3], PL, rhoR, uR[3], PR;
          double rhoL_dash, uL_dash[3], PL_dash, rhoR_dash, uR_dash[3], PR_dash;
          double rhosol, usol[3], Psol;
          double mflux, pflux[3], Eflux;
          double dm, dp[3], dE;
          int flag;

          if (iy != 0 && iz != 0) {
            Cell &rcell = cells[ix + 1][iy][iz];
            // x direction
            rhoL = lcell._rho;
            uL[0] = lcell._u[0];
            uL[1] = lcell._u[1];
            uL[2] = lcell._u[2];
            PL = lcell._P;
            rhoR = rcell._rho;
            uR[0] = rcell._u[0];
            uR[1] = rcell._u[1];
            uR[2] = rcell._u[2];
            PR = rcell._P;

            rhoL_dash = rhoL + 0.5 * CELLSIZE_X * lcell._grad_rho[0];
            uL_dash[0] = uL[0] + 0.5 * CELLSIZE_X * lcell._grad_u[0][0];
            uL_dash[1] = uL[1] + 0.5 * CELLSIZE_X * lcell._grad_u[1][0];
            uL_dash[2] = uL[2] + 0.5 * CELLSIZE_X * lcell._grad_u[2][0];
            PL_dash = PL + 0.5 * CELLSIZE_X * lcell._grad_P[0];
            rhoR_dash = rhoR - 0.5 * CELLSIZE_X * rcell._grad_rho[0];
            uR_dash[0] = uR[0] - 0.5 * CELLSIZE_X * rcell._grad_u[0][0];
            uR_dash[1] = uR[1] - 0.5 * CELLSIZE_X * rcell._grad_u[1][0];
            uR_dash[2] = uR[2] - 0.5 * CELLSIZE_X * rcell._grad_u[2][0];
            PR_dash = PR - 0.5 * CELLSIZE_X * rcell._grad_P[0];

            flag = solver.solve(rhoL_dash, uL_dash[0], PL_dash, rhoR_dash,
                                uR_dash[0], PR_dash, rhosol, usol[0], Psol);
            if (flag < 0) {
              usol[1] = uL_dash[1];
              usol[2] = uL_dash[2];
            } else {
              usol[1] = uR_dash[1];
              usol[2] = uR_dash[2];
            }

            mflux = rhosol * usol[0];
            pflux[0] = rhosol * usol[0] * usol[0] + Psol;
            pflux[1] = rhosol * usol[1] * usol[0];
            pflux[2] = rhosol * usol[2] * usol[0];
            Eflux = (Psol / (GAMMA - 1.) +
                     0.5 * rhosol * (usol[0] * usol[0] + usol[1] * usol[1] +
                                     usol[2] * usol[2])) *
                        usol[0] +
                    Psol * usol[0];

            dm = DT_CELL_AREA_X * mflux;
            dp[0] = DT_CELL_AREA_X * pflux[0];
            dp[1] = DT_CELL_AREA_X * pflux[1];
            dp[2] = DT_CELL_AREA_X * pflux[2];
            dE = DT_CELL_AREA_X * Eflux;

            lcell.lock();
            lcell._m -= dm;
            lcell._p[0] -= dp[0];
            lcell._p[1] -= dp[1];
            lcell._p[2] -= dp[2];
            lcell._E -= dE;
            lcell.unlock();

            rcell.lock();
            rcell._m += dm;
            rcell._p[0] += dp[0];
            rcell._p[1] += dp[1];
            rcell._p[2] += dp[2];
            rcell._E += dE;
            rcell.unlock();
          }

          if (ix != 0 && iz != 0) {
            Cell &rcell = cells[ix][iy + 1][iz];
            // y direction
            rhoL = lcell._rho;
            uL[0] = lcell._u[0];
            uL[1] = lcell._u[1];
            uL[2] = lcell._u[2];
            PL = lcell._P;
            rhoR = rcell._rho;
            uR[0] = rcell._u[0];
            uR[1] = rcell._u[1];
            uR[2] = rcell._u[2];
            PR = rcell._P;

            rhoL_dash = rhoL + 0.5 * CELLSIZE_Y * lcell._grad_rho[1];
            uL_dash[0] = uL[0] + 0.5 * CELLSIZE_Y * lcell._grad_u[0][1];
            uL_dash[1] = uL[1] + 0.5 * CELLSIZE_Y * lcell._grad_u[1][1];
            uL_dash[2] = uL[2] + 0.5 * CELLSIZE_Y * lcell._grad_u[2][1];
            PL_dash = PL + 0.5 * CELLSIZE_Y * lcell._grad_P[1];
            rhoR_dash = rhoR - 0.5 * CELLSIZE_Y * rcell._grad_rho[1];
            uR_dash[0] = uR[0] - 0.5 * CELLSIZE_Y * rcell._grad_u[0][1];
            uR_dash[1] = uR[1] - 0.5 * CELLSIZE_Y * rcell._grad_u[1][1];
            uR_dash[2] = uR[2] - 0.5 * CELLSIZE_Y * rcell._grad_u[2][1];
            PR_dash = PR - 0.5 * CELLSIZE_Y * rcell._grad_P[1];

            flag = solver.solve(rhoL_dash, uL_dash[1], PL_dash, rhoR_dash,
                                uR_dash[1], PR_dash, rhosol, usol[1], Psol);
            if (flag < 0) {
              usol[0] = uL_dash[0];
              usol[2] = uL_dash[2];
            } else {
              usol[0] = uR_dash[0];
              usol[2] = uR_dash[2];
            }

            mflux = rhosol * usol[1];
            pflux[0] = rhosol * usol[0] * usol[1];
            pflux[1] = rhosol * usol[1] * usol[1] + Psol;
            pflux[2] = rhosol * usol[2] * usol[1];
            Eflux = (Psol / (GAMMA - 1.) +
                     0.5 * rhosol * (usol[0] * usol[0] + usol[1] * usol[1] +
                                     usol[2] * usol[2])) *
                        usol[1] +
                    Psol * usol[1];

            dm = DT_CELL_AREA_Y * mflux;
            dp[0] = DT_CELL_AREA_Y * pflux[0];
            dp[1] = DT_CELL_AREA_Y * pflux[1];
            dp[2] = DT_CELL_AREA_Y * pflux[2];
            dE = DT_CELL_AREA_Y * Eflux;

            lcell.lock();
            lcell._m -= dm;
            lcell._p[0] -= dp[0];
            lcell._p[1] -= dp[1];
            lcell._p[2] -= dp[2];
            lcell._E -= dE;
            lcell.unlock();

            rcell.lock();
            rcell._m += dm;
            rcell._p[0] += dp[0];
            rcell._p[1] += dp[1];
            rcell._p[2] += dp[2];
            rcell._E += dE;
            rcell.unlock();
          }

          if (ix != 0 && iy != 0) {
            Cell &rcell = cells[ix][iy][iz + 1];
            // z direction
            rhoL = lcell._rho;
            uL[0] = lcell._u[0];
            uL[1] = lcell._u[1];
            uL[2] = lcell._u[2];
            PL = lcell._P;
            rhoR = rcell._rho;
            uR[0] = rcell._u[0];
            uR[1] = rcell._u[1];
            uR[2] = rcell._u[2];
            PR = rcell._P;

            rhoL_dash = rhoL + 0.5 * CELLSIZE_Z * lcell._grad_rho[2];
            uL_dash[0] = uL[0] + 0.5 * CELLSIZE_Z * lcell._grad_u[0][2];
            uL_dash[1] = uL[1] + 0.5 * CELLSIZE_Z * lcell._grad_u[1][2];
            uL_dash[2] = uL[2] + 0.5 * CELLSIZE_Z * lcell._grad_u[2][2];
            PL_dash = PL + 0.5 * CELLSIZE_Z * lcell._grad_P[2];
            rhoR_dash = rhoR - 0.5 * CELLSIZE_Z * rcell._grad_rho[2];
            uR_dash[0] = uR[0] - 0.5 * CELLSIZE_Z * rcell._grad_u[0][2];
            uR_dash[1] = uR[1] - 0.5 * CELLSIZE_Z * rcell._grad_u[1][2];
            uR_dash[2] = uR[2] - 0.5 * CELLSIZE_Z * rcell._grad_u[2][2];
            PR_dash = PR - 0.5 * CELLSIZE_Z * rcell._grad_P[2];

            flag = solver.solve(rhoL_dash, uL_dash[2], PL_dash, rhoR_dash,
                                uR_dash[2], PR_dash, rhosol, usol[2], Psol);
            if (flag < 0) {
              usol[0] = uL_dash[0];
              usol[1] = uL_dash[1];
            } else {
              usol[0] = uR_dash[0];
              usol[1] = uR_dash[1];
            }

            mflux = rhosol * usol[2];
            pflux[0] = rhosol * usol[0] * usol[2];
            pflux[1] = rhosol * usol[1] * usol[2];
            pflux[2] = rhosol * usol[2] * usol[2] + Psol;
            Eflux = (Psol / (GAMMA - 1.) +
                     0.5 * rhosol * (usol[0] * usol[0] + usol[1] * usol[1] +
                                     usol[2] * usol[2])) *
                        usol[2] +
                    Psol * usol[2];

            dm = DT_CELL_AREA_Z * mflux;
            dp[0] = DT_CELL_AREA_Z * pflux[0];
            dp[1] = DT_CELL_AREA_Z * pflux[1];
            dp[2] = DT_CELL_AREA_Z * pflux[2];
            dE = DT_CELL_AREA_Z * Eflux;

            lcell.lock();
            lcell._m -= dm;
            lcell._p[0] -= dp[0];
            lcell._p[1] -= dp[1];
            lcell._p[2] -= dp[2];
            lcell._E -= dE;
            lcell.unlock();

            rcell.lock();
            rcell._m += dm;
            rcell._p[0] += dp[0];
            rcell._p[1] += dp[1];
            rcell._p[2] += dp[2];
            rcell._E += dE;
            rcell.unlock();
          }
        }
      }
    }

#if POTENTIAL == POTENTIAL_DISC
    for (unsigned int i = 0; i < NCELL_TOTAL; ++i) {
      double reduction_factor = 1.;
      if (istep * DT < 5. * 48.) {
        reduction_factor = istep * DT / 5. / 48.;
      }
      double a = reduction_factor * 2. * M_PI * 10. *
                 std::tanh(cells[i]._midpoint[0] * 0.01);
      a *= G;
      cells[i]._a[0] = a;
      cells[i]._p[0] -= 0.5 * DT * a * cells[i]._m;
    }
#elif POTENTIAL == POTENTIAL_DISC_SUP
    for (unsigned int i = 0; i < NCELL_TOTAL; ++i) {
      double reduction_factor = 1.;
      if (istep * DT < DISC_SUP_GROWTH_TIME) {
        reduction_factor = istep * DT / DISC_SUP_GROWTH_TIME;
      }
      double a;
      if (cells[i]._midpoint[0] > -300. && cells[i]._midpoint[0] < 300.) {
        a = 2. * M_PI * 10. * std::tanh(cells[i]._midpoint[0] * 0.01);
      } else if (cells[i]._midpoint[0] > -350. &&
                 cells[i]._midpoint[0] < 350.) {
        a = 2. * M_PI * 10. * std::tanh(cells[i]._midpoint[0] * 0.01) *
            (0.5 +
             0.5 * std::cos(M_PI * (std::abs(cells[i]._midpoint[0]) - 300.) *
                            0.02));
      } else {
        a = 0.;
      }
      a *= reduction_factor * G;
      cells[i]._a[0] = a;
      cells[i]._p[0] -= 0.5 * DT * a * cells[i]._m;
    }
#endif
  }

  timer.stop();
  std::cout << "Total program time: " << timer.value() << " s." << std::endl;
  return 0;
}
