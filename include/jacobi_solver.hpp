#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include "grid.hpp"
#include <mpi.h>

/**
 * @file jacobi_solver.hpp
 * @brief Jacobi iterative solver for the Laplace equation.
 *
 * The solver performs the following steps each iteration:
 *   1. Data exchange with neighbouring MPI ranks.
 *   2. Update all owned interior points using the four-point Jacobi
 *      updating rule (parallelised with OpenMP over rows).
 *   3. Compute the local contribution to the global error norm.
 *   4. Allreduce to obtain the global error and check convergence.
 *   5. Swap U and U_new buffers.
 */

/// Struct to hold the solver result
struct SolverResult
{
    int    iterations;   ///< Number of iterations performed
    double error;        ///< Final global error norm
    bool   converged;    ///< Whether tolerance was reached or not
};

/**
 * @brief Run the Jacobi method on the given grid.
 *
 * @param g          Grid.
 * @param f          Forcing term f(x, y).
 * @param bc_bottom  Dirichlet BC at y = 0.
 * @param bc_top     Dirichlet BC at y = 1.
 * @param bc_left    Dirichlet BC at x = 0.
 * @param bc_right   Dirichlet BC at x = 1.
 * @param tol        Convergence tolerance on the global error norm.
 * @param max_iter   Maximum number of iterations.
 * @param comm       MPI communicator.
 * @return           SolverResult with iteration count, final error and convergence flag.
 */
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