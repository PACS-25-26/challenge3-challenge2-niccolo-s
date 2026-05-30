#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP

#include "grid.hpp"
#include <mpi.h>

/**
 * @file mpi_utils.hpp
 * @brief MPI communication utilities for the Jacobi solver.
 *
 * Two responsibilities:
 *   1. Data exchange: at the start of each Jacobi iteration, each rank
 *      sends its top/bottom owned rows to its neighbours and receives
 *      their boundary rows into its ghost slots.
 *
 *   2. Global convergence check: each rank computes a local error norm;
 *      MPI_Allreduce combines them so every rank knows the global error.
 */

/**
 * @brief Exchange ghost rows with neighbouring MPI ranks.
 *
 * Each rank simultaneously:
 *   - sends its first owned row (local index 1) to rank_above
 *   - receives into ghost top (local index 0) from rank_above
 *   - sends its last owned row (local index local_n) to rank_below
 *   - receives into ghost bottom (local index local_n+1) from rank_below
 *
 * MPI_Sendrecv is used for each direction to avoid deadlocks.
 * Ranks at the domain boundary (rank_above == -1 or rank_below == -1)
 * skip the corresponding exchange. Their ghost slots are never read
 * by the Jacobi update anyway (boundary rows are fixed).
 *
 * @param g     Grid whose U buffer is updated.
 * @param comm  MPI communicator (defaulted to MPI_COMM_WORLD).
 */
void data_exchange(Grid& g, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Compute the global L2 error norm between U and U_new to assess convergence.
 *
 * Each rank computes the sum of squared differences over its owned
 * interior rows between two consecutive iterations. MPI_Allreduce 
 * sums these contributions, then every rank takes sqrt(h * total_sum).
 *
 * @param g          Grid (used for h, local_n, n).
 * @param local_sum  Sum of (U_new - U)^2 over this rank's interior points.
 * @param comm       MPI communicator (defaulted to MPI_COMM_WORLD).
 * @return           Global error  e = sqrt( h * sum_{i,j} (U_new - U)^2 ).
 */
double check_convergence(const Grid& g, double local_sum, MPI_Comm comm = MPI_COMM_WORLD);

#endif // MPI_UTILS_HPP