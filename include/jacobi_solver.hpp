#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include "grid.hpp"
#include <mpi.h>

/**
 * @file jacobi_solver.hpp
 * @brief Jacobi iterative solver for the Laplace equation.
 *
 * Supports Dirichlet, Neumann, and Robin boundary conditions.
 * Neumann and Robin BCs are handled via the ghost node method,
 * which gives second-order accuracy at boundary nodes.
 * 
 * The solver performs the following steps each iteration:
 *   1. Data exchange with neighbouring MPI ranks.
 *   2. Update all owned interior nodes using the four-point Jacobi
 *      updating rule (parallelised with OpenMP over rows) and possibly
 *      Neumann/Robin nodes according to the ghost-node method.
 *   3. Compute the local contribution to the global increment error.
 *   4. Allreduce to obtain the global error and check convergence.
 *   5. Swap U and U_new buffers and re-enforce Dirichlet BCs.
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
 * @param bc_bottom  BC at y = 0.
 * @param bc_top     BC at y = 1.
 * @param bc_left    BC at x = 0.
 * @param bc_right   BC at x = 1.
 * @param tol        Convergence tolerance on the global error norm.
 * @param max_iter   Maximum number of iterations.
 * @param comm       MPI communicator.
 * @return           SolverResult with iteration count, final error and convergence flag.
 */
SolverResult jacobi_solve(Grid& g,
                           const Field& f,
                           const BoundaryCondition& bc_top,
                           const BoundaryCondition& bc_bottom,
                           const BoundaryCondition& bc_left,
                           const BoundaryCondition& bc_right,
                           double tol,
                           int max_iter,
                           MPI_Comm comm = MPI_COMM_WORLD);

#endif // JACOBI_SOLVER_HPP