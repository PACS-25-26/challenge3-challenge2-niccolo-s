#ifndef GRID_HPP
#define GRID_HPP

#include <vector>
#include <functional>
#include <stdexcept>


// Function-type alias that takes two doubles and returns a double (for BCs, forcing term, ...)
using Field = std::function<double(double, double)>;


class Grid
{
    // Grid parameters
    int    n_;
    double h_;

    // MPI parameters
    int rank_;
    int size_;

    // Local ownership
    int global_row_begin_;
    int global_row_end_;
    int local_n_;

    // Neighbour ranks (-1 = no neighbour / domain boundary)
    int rank_above_;
    int rank_below_;

    // Solution buffers  (size = (local_n+2) * n)
    std::vector<double> U_;
    std::vector<double> U_new_;

    // Coordinates
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;


public:

    Grid(int n, int rank, int size,
         const Field& bc_top,
         const Field& bc_bottom,
         const Field& bc_left,
         const Field& bc_right);

    
    // 2D element access
    double& operator()(int r, int c)       { return U_[r * n_ + c]; }
    double  operator()(int r, int c) const { return U_[r * n_ + c]; }

    // Read/write access to U_new (next iteration).
    double& new_(int r, int c)       { return U_new_[r * n_ + c]; }
    double  new_(int r, int c) const { return U_new_[r * n_ + c]; }

    // Swap U and U_new at the end of each Jacobi iteration
    void swap_buffers() { std::swap(U_, U_new_); }

    // Boundary conditions
    void apply_boundary_conditions(const Field& bc_top,
                                   const Field& bc_bottom,
                                   const Field& bc_left,
                                   const Field& bc_right);

    // Pointer to the first element of local row r in U.
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
    double y_coord(int r) const { return y_coords_[r]; }

    // Static helper: row range for any rank
    static void compute_row_range(int n, int rank, int size,
                                  int& begin, int& end);

};

#endif // GRID_HPP
