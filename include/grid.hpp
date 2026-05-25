#ifndef GRID_HPP
#define GRID_HPP

#include <vector>
#include <functional>
#include <stdexcept>

/**
 * @file grid.hpp
 * @brief Grid class for MPI domain decomposition of the Laplace solver.
 *
 * The full n*n grid is decomposed by rows among MPI ranks.
 * Each rank owns a contiguous strip of rows, plus two ghost rows
 * (one above, one below) used for exchange with neighbours in Jacobi iterations.
 *
 * Local storage layout (flat vector, row-major):
 *
 *   local index 0        : ghost row top    (data from rank_above)
 *   local index 1        : first owned row
 *   ...
 *   local index local_n  : last  owned row
 *   local index local_n+1: ghost row bottom (data from rank_below)
 *
 * 2D access via operator():
 *   grid(r, c)       ->  U[r * n + c]      (solution at current iteration)
 *   grid.new_(r, c)  ->  U_new[r * n + c]  (solution at next iteration)
 */


/// Function-type alias that takes two doubles and returns a double (for BCs, forcing term, ...)
using Field = std::function<double(double, double)>;


class Grid
{
    // Grid parameters
    int    n_;           ///< Global grid size
    double h_;           ///< Mesh spacing h = 1/(n-1)

    // MPI parameters
    int rank_;
    int size_;

    // Local ownership
    int global_row_begin_;  ///< Index of first row (included)
    int global_row_end_;    ///< Index of last row (included)
    int local_n_;           ///< Number of owned rows

    // Neighbour ranks (-1 = no neighbour / domain boundary)
    int rank_above_;
    int rank_below_;

    // Solution buffers  (size = (local_n+2) * n)
    std::vector<double> U_;      ///< Current iteration (includes ghost rows)
    std::vector<double> U_new_;  ///< Next iteration, same layout as U, ghost slots unused

    // Coordinates
    std::vector<double> x_coords_; ///< x-coords for all n columns
    std::vector<double> y_coords_; ///< y-coords for owned rows (local_n entries)


public:

    /**
     * @brief Grid constructor. Build and initialise the grid for the calling MPI rank.
     *
     * Computes the row decomposition so that rows are distributed as evenly
     * as possible (remainder rows are given to the first ranks one by one).
     * Allocates U and U_new, fills boundary rows with the Dirichlet BC
     * functions, and sets all interior entries to zero.
     *
     * @param n          Global grid size (n points per side).
     * @param rank       MPI rank of the calling process.
     * @param size       Total number of MPI processes.
     * @param bc_top     Dirichlet BC at y = 1 (top boundary, global row 0).
     * @param bc_bottom  Dirichlet BC at y = 0 (bottom boundary, global row n-1).
     * @param bc_left    Dirichlet BC at x = 0 (column 0).
     * @param bc_right   Dirichlet BC at x = 1 (column n-1).
     */
    Grid(int n, int rank, int size,
         const Field& bc_top,
         const Field& bc_bottom,
         const Field& bc_left,
         const Field& bc_right);

    
    // 2D element access

    /// Read/write access to U (current iteration).
    /// r is the local row index (0 = ghost top; 1, ..., local_n = owned; local_n+1 = ghost bottom).
    double& operator()(int r, int c)       { return U_[r * n_ + c]; }
    double  operator()(int r, int c) const { return U_[r * n_ + c]; }

    /// Read/write access to U_new (next iteration).
    double& new_(int r, int c)       { return U_new_[r * n_ + c]; }
    double  new_(int r, int c) const { return U_new_[r * n_ + c]; }

    // Swap U and U_new at the end of each Jacobi iteration
    void swap_buffers() { std::swap(U_, U_new_); }

    // Boundary conditions
    /**
     * @brief Apply Dirichlet BCs to all owned boundary rows/columns.
     */
    void apply_boundary_conditions(const Field& bc_top,
                                   const Field& bc_bottom,
                                   const Field& bc_left,
                                   const Field& bc_right);

    // Raw buffer access (needed by MPI send/recv in mpi_utils)

    /// Pointer to the first element of local row r in U.
    double*       row_ptr(int r)       { return U_.data() + r * n_; }
    const double* row_ptr(int r) const { return U_.data() + r * n_; }

    // Getters
    int    n()                const { return n_; }
    double h()                const { return h_; }
    int    rank()             const { return rank_; }
    int    size()             const { return size_; }
    int    local_n()          const { return local_n_; }
    int    global_row_begin() const { return global_row_begin_; }
    int    global_row_end()   const { return global_row_end_; }
    int    rank_above()       const { return rank_above_; }
    int    rank_below()       const { return rank_below_; }

    double x_coord(int j) const { return x_coords_[j]; }
    double y_coord(int r) const { return y_coords_[r]; } ///< r is local owned index (0, ..., local_n-1)

    // Static helper: row range for any rank
    /**
     * @brief Compute the row range [begin, end] (inclusive, 0-based) for
     *        a given rank when distributing n rows over `size` processes.
     */
    static void compute_row_range(int n, int rank, int size,
                                  int& begin, int& end);

};

#endif // GRID_HPP
