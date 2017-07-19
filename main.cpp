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

/**
 * @file main.cpp
 *
 * @brief 3D hydro solver.
 *
 * For clarity, each function indicates the SI quantity represented by the input
 * and output variables. The only relevant SI quantities in this code are length
 * ([L]), time ([T]), and mass ([M]).
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

#include "RiemannSolver.hpp"
#include "Timer.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <vector>

/// parameter definitions

// supported problem setups
#define PROBLEM_SOD_SHOCK 0
#define PROBLEM_DISC_PATCH 1

// supported boundary condition types
#define BOUNDARIES_PERIODIC 0
#define BOUNDARIES_OPEN 1
#define BOUNDARIES_REFLECTIVE 2

// supported equations of state
#define EOS_IDEAL 0
#define EOS_ISOTHERMAL 1

// supported external potentials
#define POTENTIAL_NONE 0
#define POTENTIAL_DISC 1

/// customizable parameters

// problem we want to solve
#define PROBLEM PROBLEM_SOD_SHOCK

// number of cells in each spatial dimension
#define NCELL_X 100
#define NCELL_Y 10
#define NCELL_Z 10

// adiabatic index
#define GAMMA (5. / 3.)

// slope limiter constant (1/2 is the most conservative value, larger values are
// more accurate but less stable)
#define BETA 0.5

// type of boundary conditions to use in each dimension
#define BOUNDARIES_X BOUNDARIES_PERIODIC
#define BOUNDARIES_Y BOUNDARIES_PERIODIC
#define BOUNDARIES_Z BOUNDARIES_PERIODIC

// equation of state to use
#define EOS EOS_ISOTHERMAL

// isothermal thermal energy (if EOS_ISOTHERMAL is selected)
#define ISOTHERMAL_U 20.26785

// value for the gravitational constant G (if a potential is used)
#define G 4.30097e-03

/// end of customizable parameters

// sanity checks
#ifndef PROBLEM
#error "No problem selected!"
#endif

#ifndef BOUNDARIES_X
#error "No boundary condition chosen in the x direction!"
#endif
#ifndef BOUNDARIES_Y
#error "No boundary condition chosen in the y direction!"
#endif
#ifndef BOUNDARIES_Z
#error "No boundary condition chosen in the z direction!"
#endif

#ifndef EOS
#error "No equation of state chosen!"
#endif

#if PROBLEM == PROBLEM_SOD_SHOCK
/// sod shock setup <<<
#define XMIN 0.
#define XMAX 1.
#define YMIN 0.
#define YMAX 1.
#define ZMIN 0.
#define ZMAX 1.

#define DT 0.001
#define NSTEP 100
#define SNAPSTEP 10
#define SNAP_PREFIX "sod_"

#define POTENTIAL POTENTIAL_NONE
/// >>>
#elif PROBLEM == PROBLEM_DISC_PATCH
/// disc patch setup <<<
#define XMIN (-200.)
#define XMAX 200.
#define YMIN (-200.)
#define YMAX 200.
#define ZMIN (-200.)
#define ZMAX 200.

#define DT 0.01
#define NSTEP 1000000
#define SNAPSTEP 10000
#define SNAP_PREFIX "disc_"

#define POTENTIAL POTENTIAL_DISC
/// >>>
#else
#error "Unknown problem chosen!"
#endif

/// precomputed geometrical variables
#define BOXSIZE_X (XMAX - XMIN)
#define BOXSIZE_Y (YMAX - YMIN)
#define BOXSIZE_Z (ZMAX - ZMIN)
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

// isothermal soundspeed (if EOS_ISOTHERMAL is chosen)
#define ISOTHERMAL_C (ISOTHERMAL_U * (GAMMA - 1.))

// precomputed quantities involving the adiabatic index
#define GAMMA_MINUS_ONE (GAMMA - 1.)
#define ONE_OVER_GAMMA_MINUS_ONE (1. / (GAMMA - 1.))

/**
 * @brief Get the pressure for the given total energy, density and velocity.
 *
 * @param E Total energy ([M] [L]^2 [T]^-2).
 * @param rho Density ([M] [L]^-3).
 * @param u Velocity (vector, [M] [T]^-1).
 * @return Pressure ([M] [L]^-1 [T]^-2).
 */
#if EOS == EOS_IDEAL
#define get_pressure(E, rho, u)                                                \
  GAMMA_MINUS_ONE *(E * CELL_VOLUME_INV -                                      \
                    0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]))
#elif EOS == EOS_ISOTHERMAL
#define get_pressure(E, rho, u) ISOTHERMAL_C *rho;
#else
#error "No equation of state selected!"
#endif

/**
 * @brief Compute the primitive variables for the given values of the conserved
 * variables.
 *
 * @param rho Density value to compute ([M] [L]^-3).
 * @param u Velocity value to compute (vector, [M] [T]^-1).
 * @param P Pressure value to compute ([M] [L]^-1 [T]^-2).
 * @param m Mass value ([M]).
 * @param p Momentum value (vector, [M] [L] [T]^-1).
 * @param E Energy value ([M] [L]^2 [T]^-2).
 */
#define compute_primitive_variables(rho, u, P, m, p, E)                        \
  rho = m * CELL_VOLUME_INV;                                                   \
  const double minv = 1. / m;                                                  \
  u[0] = p[0] * minv;                                                          \
  u[1] = p[1] * minv;                                                          \
  u[2] = p[2] * minv;                                                          \
  P = get_pressure(E, rho, u);

/**
 * @brief Copy the primitive variables from the given cell to the given other
 * cell.
 *
 * @param from Cell to copy from.
 * @param to Cell to copy to.
 */
#define copy_primitives(from, to)                                              \
  to._rho = from._rho;                                                         \
  to._u[0] = from._u[0];                                                       \
  to._u[1] = from._u[1];                                                       \
  to._u[2] = from._u[2];                                                       \
  to._P = from._P;

/**
 * @brief Compute the gradient for the given quantity, based on the given value
 * of the quantity in the cell and the neighbouring cells.
 *
 * The quantity is expressed in units [Q].
 *
 * @param grad_phi Gradient to compute ([Q] [L]^-1).
 * @param phi Value of the quantity in the cell ([Q]).
 * @param phi_left Value of the quantity in the neighbouring cell with a smaller
 * x coordinate ([Q]).
 * @param phi_right Value of the quantity in the neighbouring cell with a larger
 * x coordinate ([Q]).
 * @param phi_front Value of the quantity in the neighbouring cell with a
 * smaller y coordinate ([Q]).
 * @param phi_back Value of the quantity in the neighbouring cell with a larger
 * y coordinate ([Q]).
 * @param phi_bottom Value of the quantity in the neighbouring cell with a
 * smaller z coordinate ([Q]).
 * @param phi_top Value of the quantity in the neighbouring cell with a larger
 * z coordinate ([Q]).
 */
#define compute_gradient(grad_phi, phi, phi_left, phi_right, phi_front,        \
                         phi_back, phi_bottom, phi_top)                        \
  {                                                                            \
    grad_phi[0] = 0.5 * CELLSIZE_X_INV * (phi_right - phi_left);               \
    grad_phi[1] = 0.5 * CELLSIZE_Y_INV * (phi_back - phi_front);               \
    grad_phi[2] = 0.5 * CELLSIZE_Z_INV * (phi_top - phi_bottom);               \
                                                                               \
    /* Apply a cell wide slope limiter (Hopkins, 2015) */                      \
    double alpha, phi_ngb_max, phi_ext_max, phi_ngb_min, phi_ext_min;          \
    phi_ngb_max = std::max(phi_left, phi_right);                               \
    phi_ngb_max = std::max(phi_ngb_max, phi_front);                            \
    phi_ngb_max = std::max(phi_ngb_max, phi_back);                             \
    phi_ngb_max = std::max(phi_ngb_max, phi_bottom);                           \
    phi_ngb_max = std::max(phi_ngb_max, phi_top);                              \
    phi_ngb_max -= phi;                                                        \
    phi_ngb_min = std::min(phi_left, phi_right);                               \
    phi_ngb_min = std::min(phi_ngb_min, phi_front);                            \
    phi_ngb_min = std::min(phi_ngb_min, phi_back);                             \
    phi_ngb_min = std::min(phi_ngb_min, phi_bottom);                           \
    phi_ngb_min = std::min(phi_ngb_min, phi_top);                              \
    phi_ngb_min -= phi;                                                        \
    phi_ext_max = 0.5 * CELLSIZE_X * std::max(grad_phi[0], -grad_phi[0]);      \
    phi_ext_max = std::max(phi_ext_max, 0.5 * CELLSIZE_Y * grad_phi[1]);       \
    phi_ext_max = std::max(phi_ext_max, -0.5 * CELLSIZE_Y * grad_phi[1]);      \
    phi_ext_max = std::max(phi_ext_max, 0.5 * CELLSIZE_Z * grad_phi[2]);       \
    phi_ext_max = std::max(phi_ext_max, -0.5 * CELLSIZE_Z * grad_phi[2]);      \
    phi_ext_min = 0.5 * CELLSIZE_X * std::min(grad_phi[0], -grad_phi[0]);      \
    phi_ext_min = std::min(phi_ext_min, 0.5 * CELLSIZE_Y * grad_phi[1]);       \
    phi_ext_min = std::min(phi_ext_min, -0.5 * CELLSIZE_Y * grad_phi[1]);      \
    phi_ext_min = std::min(phi_ext_min, 0.5 * CELLSIZE_Z * grad_phi[2]);       \
    phi_ext_min = std::min(phi_ext_min, -0.5 * CELLSIZE_Z * grad_phi[2]);      \
                                                                               \
    alpha = std::min(1., BETA * std::min(phi_ngb_max / phi_ext_max,            \
                                         phi_ngb_min / phi_ext_min));          \
    grad_phi[0] *= alpha;                                                      \
    grad_phi[1] *= alpha;                                                      \
    grad_phi[2] *= alpha;                                                      \
  }

/**
 * @brief Copy the gradients from the given cell into the other given cell.
 *
 * @param from Cell to copy from.
 * @param to Cell to copy to.
 */
#define copy_gradients(from, to)                                               \
  to._grad_rho[0] = from._grad_rho[0];                                         \
  to._grad_rho[1] = from._grad_rho[1];                                         \
  to._grad_rho[2] = from._grad_rho[2];                                         \
  to._grad_u[0][0] = from._grad_u[0][0];                                       \
  to._grad_u[0][1] = from._grad_u[0][1];                                       \
  to._grad_u[0][2] = from._grad_u[0][2];                                       \
  to._grad_u[1][0] = from._grad_u[1][0];                                       \
  to._grad_u[1][1] = from._grad_u[1][1];                                       \
  to._grad_u[1][2] = from._grad_u[1][2];                                       \
  to._grad_u[2][0] = from._grad_u[2][0];                                       \
  to._grad_u[2][1] = from._grad_u[2][1];                                       \
  to._grad_u[2][2] = from._grad_u[2][2];                                       \
  to._grad_P[0] = from._grad_P[0];                                             \
  to._grad_P[1] = from._grad_P[1];                                             \
  to._grad_P[2] = from._grad_P[2];

/**
 * @brief Predict the given primitive variables forward in time for half a
 * timestep, based on the given gradients.
 *
 * @param rho Density value to update ([M] [L]^-3).
 * @param u Velocity value to update (vector, [L] [T]^-1).
 * @param P Pressure value to update ([M] [L]^-1 [T]^-2).
 * @param grad_rho Gradient of the density to use (vector, [M] [L]^-4).
 * @param grad_u Gradient of the velocity to use (tensor, [T]^-1).
 * @param grad_P Gradient of the pressure to use (vector, [M] [L]^-2 [T]^-2).
 */
#define do_half_step_prediction(rho, u, P, grad_rho, grad_u, grad_P)           \
  const double old_rho = rho;                                                  \
  const double old_u[3] = {u[0], u[1], u[2]};                                  \
  const double old_P = P;                                                      \
                                                                               \
  const double rho_inv = 1. / rho;                                             \
  const double div_u = grad_u[0][0] + grad_u[1][1] + grad_u[2][2];             \
                                                                               \
  rho -= 0.5 * DT * (old_rho * div_u + old_u[0] * grad_rho[0] +                \
                     old_u[1] * grad_rho[1] + old_u[2] * grad_rho[2]);         \
  u[0] -= 0.5 * DT * (old_u[0] * div_u + rho_inv * grad_P[0]);                 \
  u[1] -= 0.5 * DT * (old_u[1] * div_u + rho_inv * grad_P[1]);                 \
  u[2] -= 0.5 * DT * (old_u[2] * div_u + rho_inv * grad_P[2]);                 \
  P -= 0.5 * DT * (GAMMA * old_P * div_u + old_u[0] * grad_P[0] +              \
                   old_u[1] * grad_P[1] + old_u[2] * grad_P[2]);

/**
 * @brief Extrapolate the given quantity forward in time using the given
 * gradient, distance, and direction.
 *
 * The quantity is expressed in units [Q].
 *
 * @param phi Quantity to extrapolate ([Q]).
 * @param grad_phi Gradient of the quantity (vector, [Q] [L]^-1).
 * @param ds Distance over which we extrapolate ([L]).
 * @param direction Extrapolation direction (0 / 1 / 2).
 * @return Extrapolated quantity.
 */
#define extrapolate_quantity(phi, grad_phi, ds, direction)                     \
  phi + ds *grad_phi[direction]

/**
 * @brief Solve the 3D Riemann problem with the given left and right state, in
 * the given direction.
 *
 * @param rhoL Left state density ([M] [L]^-3).
 * @param uL Left state velocity (vector, [L] [T]^-1).
 * @param PL Left state pressure ([M] [L]^-1 [T]^-2).
 * @param rhoR Right state density ([M] [L]^-3).
 * @param uR Right state velocity (vector, [L] [T]^-1).
 * @param PR Right state pressure ([M] [L]^-1 [T]^-2).
 * @param rhosol Solution density to compute ([M] [L]^-3).
 * @param usol Solution velocity to compute (vector, [L] [T]^-1).
 * @param Psol Solution pressure to compute ([M] [L]^-1 [T]^-2).
 * @param direction Direction in which we solve the problem (0 / 1 / 2).
 */
#define solve_3d_riemann_problem(rhoL, uL, PL, rhoR, uR, PR, rhosol, usol,     \
                                 Psol, direction)                              \
  {                                                                            \
    const int flag =                                                           \
        solver.solve(rhoL, uL[direction], PL, rhoR, uR[direction], PR, rhosol, \
                     usol[direction], Psol);                                   \
    if (flag < 0) {                                                            \
      usol[(direction + 1) % 3] = uL[(direction + 1) % 3];                     \
      usol[(direction + 2) % 3] = uL[(direction + 2) % 3];                     \
    } else {                                                                   \
      usol[(direction + 1) % 3] = uR[(direction + 1) % 3];                     \
      usol[(direction + 2) % 3] = uR[(direction + 2) % 3];                     \
    }                                                                          \
  }

/**
 * @brief Get the fluxes between the given left and right state.
 *
 * @param mflux Mass flux to compute ([M] [L]^-2 [T]^-1).
 * @param pflux Momentum flux to compute ([M] [L]^-1 [T]^-2).
 * @param Eflux Energy flux to compute ([M] [T]^-3).
 * @param rhoL Left state density ([M] [L]^-3).
 * @param uL Left state velocity (vector, [L] [T]^-1).
 * @param PL Left state pressure ([M] [L]^-1 [T]^-2).
 * @param rhoR Right state density ([M] [L]^-3).
 * @param uR Right state velocity (vector, [L] [T]^-1).
 * @param PR Right state pressure ([M] [L]^-1 [T]^-2).
 * @param grad_rhoL Left state density gradient (vector, [M] [L]^-4).
 * @param grad_uL Left state velocity gradient (tensor, [T]^-1).
 * @param grad_PL Left state pressure gradient (vector, [M] [L]^-2 [T]^-2).
 * @param grad_rhoR Right state density gradient (vector, [M] [L]^-4).
 * @param grad_uR Right state velocity gradient (tensor, [T]^-1).
 * @param grad_PR Right state pressure gradient (vector, [M] [L]^-2 [T]^-2).
 * @param ds Distance over which the states are extrapolated in space ([L]).
 * @param direction Direction of the fluxes (0 / 1 / 2).
 */
#define get_fluxes(mflux, pflux, Eflux, rhoL, uL, PL, rhoR, uR, PR, grad_rhoL, \
                   grad_uL, grad_PL, grad_rhoR, grad_uR, grad_PR, ds,          \
                   direction)                                                  \
  {                                                                            \
    const double rhoL_dash =                                                   \
        extrapolate_quantity(rhoL, grad_rhoL, ds, direction);                  \
    const double uL_dash[3] = {                                                \
        extrapolate_quantity(uL[0], grad_uL[0], ds, direction),                \
        extrapolate_quantity(uL[1], grad_uL[1], ds, direction),                \
        extrapolate_quantity(uL[2], grad_uL[2], ds, direction)};               \
    const double PL_dash = extrapolate_quantity(PL, grad_PL, ds, direction);   \
    const double rhoR_dash =                                                   \
        extrapolate_quantity(rhoR, grad_rhoR, -ds, direction);                 \
    const double uR_dash[3] = {                                                \
        extrapolate_quantity(uR[0], grad_uR[0], -ds, direction),               \
        extrapolate_quantity(uR[1], grad_uR[1], -ds, direction),               \
        extrapolate_quantity(uR[2], grad_uR[2], -ds, direction)};              \
    const double PR_dash = extrapolate_quantity(PR, grad_PR, -ds, direction);  \
                                                                               \
    double rhosol, usol[3], Psol;                                              \
    solve_3d_riemann_problem(rhoL_dash, uL_dash, PL_dash, rhoR_dash, uR_dash,  \
                             PR_dash, rhosol, usol, Psol, direction);          \
                                                                               \
    mflux = rhosol * usol[direction];                                          \
    pflux[0] = rhosol * usol[0] * usol[direction];                             \
    pflux[1] = rhosol * usol[1] * usol[direction];                             \
    pflux[2] = rhosol * usol[2] * usol[direction];                             \
    pflux[direction] += Psol;                                                  \
    Eflux = (Psol * ONE_OVER_GAMMA_MINUS_ONE +                                 \
             0.5 * rhosol * (usol[0] * usol[0] + usol[1] * usol[1] +           \
                             usol[2] * usol[2])) *                             \
                usol[direction] +                                              \
            Psol * usol[direction];                                            \
  }

/**
 * @brief Do the flux exchange assuming the active cell has a lower coordinate
 * in the given direction than the neighbouring cell.
 *
 * @param lcell Active cell.
 * @param rcell Neighbouring cell.
 * @param ds Distance between the midpoint of the active cell and the midpoint
 * of the face between the active cell and its neighbour ([L]).
 * @param Adt Surface area of the face times the system time step ([L]^2 [T]).
 * @param direction Direction of the face (0 / 1 / 2).
 */
#define do_flux_exchange_left(lcell, rcell, ds, Adt, direction)                \
  {                                                                            \
    double mflux, pflux[3], Eflux;                                             \
    get_fluxes(mflux, pflux, Eflux, lcell._rho, lcell._u, lcell._P,            \
               rcell._rho, rcell._u, rcell._P, lcell._grad_rho, lcell._grad_u, \
               lcell._grad_P, rcell._grad_rho, rcell._grad_u, rcell._grad_P,   \
               ds, direction);                                                 \
    lcell._m -= Adt * mflux;                                                   \
    lcell._p[0] -= Adt * pflux[0];                                             \
    lcell._p[1] -= Adt * pflux[1];                                             \
    lcell._p[2] -= Adt * pflux[2];                                             \
    lcell._E -= Adt * Eflux;                                                   \
  }

/**
 * @brief Do the flux exchange assuming the active cell has a higher coordinate
 * in the given direction than the neighbouring cell.
 *
 * @param lcell Active cell.
 * @param rcell Neighbouring cell.
 * @param ds Distance between the midpoint of the active cell and the midpoint
 * of the face between the active cell and its neighbour ([L]).
 * @param Adt Surface area of the face times the system time step ([L]^2 [T]).
 * @param direction Direction of the face (0 / 1 / 2).
 */
#define do_flux_exchange_right(lcell, rcell, ds, Adt, direction)               \
  {                                                                            \
    double mflux, pflux[3], Eflux;                                             \
    get_fluxes(mflux, pflux, Eflux, rcell._rho, rcell._u, rcell._P,            \
               lcell._rho, lcell._u, lcell._P, rcell._grad_rho, rcell._grad_u, \
               rcell._grad_P, lcell._grad_rho, lcell._grad_u, lcell._grad_P,   \
               ds, direction);                                                 \
    lcell._m += Adt * mflux;                                                   \
    lcell._p[0] += Adt * pflux[0];                                             \
    lcell._p[1] += Adt * pflux[1];                                             \
    lcell._p[2] += Adt * pflux[2];                                             \
    lcell._E += Adt * Eflux;                                                   \
  }

/**
 * @brief Cell.
 */
class Cell {
public:
  /*! @brief Density ([M] [L]^-3). */
  double _rho;
  /*! @brief Velocity (vector, [L] [T]^-1). */
  double _u[3];
  /*! @brief Pressure ([M] [L]^-1 [T]^-2). */
  double _P;

  /*! @brief Density gradient (vector, [M] [L]^-4). */
  double _grad_rho[3];
  /*! @brief Velocity gradient (tensor, [T]^-1). */
  double _grad_u[3][3];
  /*! @brief Pressure gradient (vector, [M] [L]^-2 [T]^-2). */
  double _grad_P[3];

  /*! @brief Mass ([M]). */
  double _m;
  /*! @brief Momentum ([M] [L] [T]^-1). */
  double _p[3];
  /*! @brief Total energy ([M] [L]^2 [T]^-2). */
  double _E;

  /*! @brief Midpoint of the cell (vector, [L]). */
  double _midpoint[3];

  /*! @brief Acceleration (vector, [L] [T]^-2). */
  double _a[3];
};

/**
 * @brief Initialize the primitive variables for the given cell.
 *
 * @param cell Cell to initialize.
 */
void init(Cell &cell) {
#if PROBLEM == PROBLEM_SOD_SHOCK
  if (cell._midpoint[0] < XMIN + 0.5 * BOXSIZE_X) {
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
#elif PROBLEM == PROBLEM_DISC_PATCH
  cell._rho = 9.64028 / 400.;
  cell._u[0] = 0.;
  cell._u[1] = 0.;
  cell._u[2] = 0.;
  cell._a[0] = 0.;
  cell._a[1] = 0.;
  cell._a[2] = 0.;
#else
#error "No problem chosen!"
#endif
}

/**
 * @brief Main program.
 *
 * @param argc Number of command line arguments passed on to the program.
 * @param argv Command line arguments passed on to the program.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {
// for reference: print out the isothermal sound speed (if we use it).
#if EOS == EOS_ISOTHERMAL
  std::cout << "ISOTHERMAL_C: " << ISOTHERMAL_C << std::endl;
#endif

  // get and print the number of shared memory threads that will be used
  int num_threads = 0;
#pragma omp parallel
  {
#pragma omp single
    { num_threads = omp_get_num_threads(); }
  }

  std::cout << "Using " << num_threads << " threads." << std::endl;

  // state the timer that will time our total program execution time
  Timer timer;
  timer.start();

  // initialize the cells
  // this allocates a huge chunk of memory
  std::vector< std::vector< std::vector< Cell > > > cells(
      NCELL_X + 2, std::vector< std::vector< Cell > >(
                       NCELL_Y + 2, std::vector< Cell >(NCELL_Z + 2)));

// initialize the cells:
// compute the cell positions, set the primitive variables, and convert them
// to conserved variables
#pragma omp parallel for default(shared) collapse(3)
  for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
    for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        Cell &cell = cells[ix][iy][iz];

        cell._midpoint[0] = XMIN + (ix - 0.5) * CELLSIZE_X;
        cell._midpoint[1] = YMIN + (iy - 0.5) * CELLSIZE_Y;
        cell._midpoint[2] = ZMIN + (iz - 0.5) * CELLSIZE_Z;

        init(cell);

#if EOS == EOS_ISOTHERMAL
        cell._P = ISOTHERMAL_C * cell._rho;
#endif

        cell._m = cell._rho * CELL_VOLUME;
        cell._p[0] = cell._m * cell._u[0];
        cell._p[1] = cell._m * cell._u[1];
        cell._p[2] = cell._m * cell._u[2];
        cell._E = cell._P * CELL_VOLUME * ONE_OVER_GAMMA_MINUS_ONE +
                  0.5 * (cell._u[0] * cell._p[0] + cell._u[1] * cell._p[1] +
                         cell._u[2] * cell._p[2]);
      }
    }
  }

  // allocate the Riemann solver
  const RiemannSolver solver(GAMMA);

  // main time integration loop
  for (unsigned int istep = 0; istep < NSTEP; ++istep) {

#if POTENTIAL == POTENTIAL_DISC
    // initialize variables used by the potential this time step
    double reduction_factor = 1.;
    if (istep * DT < 5. * 48.) {
      reduction_factor = istep * DT / 5. / 48.;
    }
#endif

// apply the gravitational potential and convert the conserved variables to
// primitive variables
#pragma omp parallel for default(shared) collapse(3)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
          Cell &cell = cells[ix][iy][iz];

#if POTENTIAL == POTENTIAL_DISC
          double a = reduction_factor * 2. * M_PI * 10. *
                     std::tanh(cell._midpoint[0] * 0.01);
          a *= G;
          cell._a[0] = a;
          cell._p[0] -= 0.5 * DT * a * cell._m;
#endif

          compute_primitive_variables(cell._rho, cell._u, cell._P, cell._m,
                                      cell._p, cell._E);
        }
      }
    }

    // check if we need to output a snapshot file
    if (istep % SNAPSTEP == 0) {
      std::cout << "step " << istep << " of " << NSTEP << std::endl;
      std::stringstream filename;
      filename << SNAP_PREFIX;
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

// the primitive variables were updated, so we need to set new values in our
// boundary ghost cells
// x direction
#pragma omp parallel for default(shared) collapse(2)
    for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        Cell &low_ghost = cells[0][iy][iz];
        Cell &high_ghost = cells[NCELL_X + 1][iy][iz];
        const Cell &low_real = cells[1][iy][iz];
        const Cell &high_real = cells[NCELL_X][iy][iz];

#if BOUNDARIES_X == BOUNDARIES_PERIODIC
        copy_primitives(high_real, low_ghost);
        copy_primitives(low_real, high_ghost);
#elif BOUNDARIES_X == BOUNDARIES_OPEN
        copy_primitives(low_real, low_ghost);
        copy_primitives(high_real, high_ghost);
#elif BOUNDARIES_X == BOUNDARIES_REFLECTIVE
        copy_primitives(low_real, low_ghost);
        copy_primitives(high_real, high_ghost);
        low_ghost._u[0] = -low_ghost._u[0];
        high_ghost._u[0] = -high_ghost._u[0];
#else
#error "Unknown boundary conditions chosen in the x direction!"
#endif
      }
    }
// y direction
#pragma omp parallel for default(shared) collapse(2)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        Cell &low_ghost = cells[ix][0][iz];
        Cell &high_ghost = cells[ix][NCELL_Y + 1][iz];
        const Cell &low_real = cells[ix][1][iz];
        const Cell &high_real = cells[ix][NCELL_Y][iz];

#if BOUNDARIES_Y == BOUNDARIES_PERIODIC
        copy_primitives(high_real, low_ghost);
        copy_primitives(low_real, high_ghost);
#elif BOUNDARIES_Y == BOUNDARIES_OPEN
        copy_primitives(low_real, low_ghost);
        copy_primitives(high_real, high_ghost);
#elif BOUNDARIES_Y == BOUNDARIES_REFLECTIVE
        copy_primitives(low_real, low_ghost);
        copy_primitives(high_real, high_ghost);
        low_ghost._u[1] = -low_ghost._u[1];
        high_ghost._u[1] = -high_ghost._u[1];
#else
#error "Unknown boundary conditions chosen in the y direction!"
#endif
      }
    }
// z direction
#pragma omp parallel for default(shared) collapse(2)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        Cell &low_ghost = cells[ix][iy][0];
        Cell &high_ghost = cells[ix][iy][NCELL_Z + 1];
        const Cell &low_real = cells[ix][iy][1];
        const Cell &high_real = cells[ix][iy][NCELL_Z];

#if BOUNDARIES_Z == BOUNDARIES_PERIODIC
        copy_primitives(high_real, low_ghost);
        copy_primitives(low_real, high_ghost);
#elif BOUNDARIES_Z == BOUNDARIES_OPEN
        copy_primitives(low_real, low_ghost);
        copy_primitives(high_real, high_ghost);
#elif BOUNDARIES_Z == BOUNDARIES_REFLECTIVE
        copy_primitives(low_real, low_ghost);
        copy_primitives(high_real, high_ghost);
        low_ghost._u[2] = -low_ghost._u[2];
        high_ghost._u[2] = -high_ghost._u[2];
#else
#error "Unknown boundary conditions chosen in the z direction!"
#endif
      }
    }

// all cells (and neighbours) have up to date primitive variables, so we can
// compute gradients for the primitive variables
#pragma omp parallel for default(shared) collapse(3)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
          Cell &cell = cells[ix][iy][iz];
          const Cell &cell_left = cells[ix - 1][iy][iz];
          const Cell &cell_right = cells[ix + 1][iy][iz];
          const Cell &cell_front = cells[ix][iy - 1][iz];
          const Cell &cell_back = cells[ix][iy + 1][iz];
          const Cell &cell_bottom = cells[ix][iy][iz - 1];
          const Cell &cell_top = cells[ix][iy][iz + 1];

          compute_gradient(cell._grad_rho, cell._rho, cell_left._rho,
                           cell_right._rho, cell_front._rho, cell_back._rho,
                           cell_bottom._rho, cell_top._rho);

          compute_gradient(cell._grad_u[0], cell._u[0], cell_left._u[0],
                           cell_right._u[0], cell_front._u[0], cell_back._u[0],
                           cell_bottom._u[0], cell_top._u[0]);

          compute_gradient(cell._grad_u[1], cell._u[1], cell_left._u[1],
                           cell_right._u[1], cell_front._u[1], cell_back._u[1],
                           cell_bottom._u[1], cell_top._u[1]);

          compute_gradient(cell._grad_u[2], cell._u[2], cell_left._u[2],
                           cell_right._u[2], cell_front._u[2], cell_back._u[2],
                           cell_bottom._u[2], cell_top._u[2]);

          compute_gradient(cell._grad_P, cell._P, cell_left._P, cell_right._P,
                           cell_front._P, cell_back._P, cell_bottom._P,
                           cell_top._P);
        }
      }
    }

// the gradients need to be updated in the boundary ghost cells
// x direction
#pragma omp parallel for default(shared) collapse(2)
    for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        Cell &low_ghost = cells[0][iy][iz];
        Cell &high_ghost = cells[NCELL_X + 1][iy][iz];
        const Cell &low_real = cells[1][iy][iz];
        const Cell &high_real = cells[NCELL_X][iy][iz];

#if BOUNDARIES_X == BOUNDARIES_PERIODIC
        copy_gradients(high_real, low_ghost);
        copy_gradients(low_real, high_ghost);
#elif BOUNDARIES_X == BOUNDARIES_OPEN
        copy_gradients(low_real, low_ghost);
        copy_gradients(high_real, high_ghost);
#elif BOUNDARIES_X == BOUNDARIES_REFLECTIVE
        copy_gradients(low_real, low_ghost);
        copy_gradients(high_real, high_ghost);
// need to add some magic
#else
#error "Unknown boundary conditions chosen in the x direction!"
#endif
      }
    }
// y direction
#pragma omp parallel for default(shared) collapse(2)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
        Cell &low_ghost = cells[ix][0][iz];
        Cell &high_ghost = cells[ix][NCELL_Y + 1][iz];
        const Cell &low_real = cells[ix][1][iz];
        const Cell &high_real = cells[ix][NCELL_Y][iz];

#if BOUNDARIES_Y == BOUNDARIES_PERIODIC
        copy_gradients(high_real, low_ghost);
        copy_gradients(low_real, high_ghost);
#elif BOUNDARIES_Y == BOUNDARIES_OPEN
        copy_gradients(low_real, low_ghost);
        copy_gradients(high_real, high_ghost);
#elif BOUNDARIES_Y == BOUNDARIES_REFLECTIVE
        copy_gradients(low_real, low_ghost);
        copy_gradients(high_real, high_ghost);
// need to add some magic
#else
#error "Unknown boundary conditions chosen in the y direction!"
#endif
      }
    }
// z direction
#pragma omp parallel for default(shared) collapse(2)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        Cell &low_ghost = cells[ix][iy][0];
        Cell &high_ghost = cells[ix][iy][NCELL_Z + 1];
        const Cell &low_real = cells[ix][iy][1];
        const Cell &high_real = cells[ix][iy][NCELL_Z];

#if BOUNDARIES_Z == BOUNDARIES_PERIODIC
        copy_gradients(high_real, low_ghost);
        copy_gradients(low_real, high_ghost);
#elif BOUNDARIES_Z == BOUNDARIES_OPEN
        copy_gradients(low_real, low_ghost);
        copy_gradients(high_real, high_ghost);
#elif BOUNDARIES_Z == BOUNDARIES_REFLECTIVE
        copy_gradients(low_real, low_ghost);
        copy_gradients(high_real, high_ghost);
// need to add some magic
#else
#error "Unknown boundary conditions chosen in the z direction!"
#endif
      }
    }

// predict the primitive variables forward in time for half a time step
#pragma omp parallel for default(shared) collapse(3)
    for (unsigned int ix = 0; ix < NCELL_X + 2; ++ix) {
      for (unsigned int iy = 0; iy < NCELL_Y + 2; ++iy) {
        for (unsigned int iz = 0; iz < NCELL_Z + 2; ++iz) {
          Cell &cell = cells[ix][iy][iz];

          do_half_step_prediction(cell._rho, cell._u, cell._P, cell._grad_rho,
                                  cell._grad_u, cell._grad_P);
        }
      }
    }

// now do the actual flux exchanges
// for better parallelization, we do every pair of cells twice, with another
// cell taking the role of active cell
// (this also means we do not have to use locking mechanisms, which seem to
//  cause problems combined with collapse(3))
#pragma omp parallel for default(shared) collapse(3)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {

          // positive x direction
          do_flux_exchange_left(cells[ix][iy][iz], cells[ix + 1][iy][iz],
                                0.5 * CELLSIZE_X, DT_CELL_AREA_X, 0);

          // negative x direction
          do_flux_exchange_right(cells[ix][iy][iz], cells[ix - 1][iy][iz],
                                 0.5 * CELLSIZE_X, DT_CELL_AREA_X, 0);

          // positive y direction
          do_flux_exchange_left(cells[ix][iy][iz], cells[ix][iy + 1][iz],
                                0.5 * CELLSIZE_Y, DT_CELL_AREA_Y, 1);

          // negative y direction
          do_flux_exchange_right(cells[ix][iy][iz], cells[ix][iy - 1][iz],
                                 0.5 * CELLSIZE_Y, DT_CELL_AREA_Y, 1);

          // positive z direction
          do_flux_exchange_left(cells[ix][iy][iz], cells[ix][iy][iz + 1],
                                0.5 * CELLSIZE_Z, DT_CELL_AREA_Z, 2);

          // negative z direction
          do_flux_exchange_right(cells[ix][iy][iz], cells[ix][iy][iz - 1],
                                 0.5 * CELLSIZE_Z, DT_CELL_AREA_Z, 2);
        }
      }
    }

// apply the next gravity kick
#if POTENTIAL == POTENTIAL_DISC
#pragma omp parallel for default(shared) collapse(3)
    for (unsigned int ix = 1; ix < NCELL_X + 1; ++ix) {
      for (unsigned int iy = 1; iy < NCELL_Y + 1; ++iy) {
        for (unsigned int iz = 1; iz < NCELL_Z + 1; ++iz) {
          Cell &cell = cells[ix][iy][iz];
          double a = reduction_factor * 2. * M_PI * 10. *
                     std::tanh(cell._midpoint[0] * 0.01);
          a *= G;
          cell._a[0] = a;
          cell._p[0] -= 0.5 * DT * a * cell._m;
        }
      }
    }
#endif
  }

  // end of the main program
  // stop the timer and output the total execution time
  timer.stop();
  std::cout << "Total program time: " << timer.value() << " s." << std::endl;
  return 0;
}
