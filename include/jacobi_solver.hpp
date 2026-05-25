#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include "grid.hpp"
#include <mpi.h>

struct SolverResult
{
    int    iterations;
    double error;
    bool   converged;
};


SolverResult jacobi(Grid& g,
                    const Field& f,
                    const Field& bc_top,
                    const Field& bc_bottom,
                    const Field& bc_left,
                    const Field& bc_right,
                    double tol,
                    int max_iter,
                    MPI_Comm comm = MPI_COMM_WORLD);

#endif // JACOBI_SOLVER_HPP