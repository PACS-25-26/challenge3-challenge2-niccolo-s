#include "grid.hpp"

#include <cmath>
#include <stdexcept>

// Helper
void Grid::compute_row_range(int n, int rank, int size, int& begin, int& end)
{
    // Distribute n rows as evenly as possible.
    // First ranks get (base+1) rows; the rest get base rows.
    const int base  = n / size;
    const int extra = n % size;

    if (rank < extra) {
        begin = rank * (base + 1);
        end   = begin + base;        // owns (base+1) rows
    } else {
        begin = extra * (base + 1) + (rank - extra) * base;
        end   = begin + base - 1;    // owns base rows
    }
}


// Constructor
Grid::Grid(int n, int rank, int size,
           const Field& bc_top,
           const Field& bc_bottom,
           const Field& bc_left,
           const Field& bc_right)
    : n_(n)
    , h_(1.0 / static_cast<double>(n - 1))
    , rank_(rank)
    , size_(size)
    , rank_above_((rank > 0)        ? rank - 1 : -1)
    , rank_below_((rank < size - 1) ? rank + 1 : -1)
    , x_coords_(n, 0.0)
    , y_coords_(0)
    , U_(0)
    , U_new_(0)
{
    if (n < 2)
        throw std::invalid_argument("Grid: n must be >= 2");
    if (size > n)
        throw std::invalid_argument("Grid: more MPI ranks than grid rows");

    // Row decomposition
    compute_row_range(n, rank, size, global_row_begin_, global_row_end_);
    local_n_ = global_row_end_ - global_row_begin_ + 1;
    y_coords_.resize(local_n_);

    // Allocate solution buffers
    U_    .assign((local_n_ + 2) * n, 0.0);
    U_new_.assign((local_n_ + 2) * n, 0.0);

    // Physical coordinates
    // x: identical for every rank
    for (int j = 0; j < n_; ++j)
        x_coords_[j] = j * h_;

    // y: only for the local_n owned rows.
    // Convention: global row 0 -> y = 1.0 (top), global row n-1 -> y = 0.0 (bottom).
    for (int r = 0; r < local_n_; ++r)
        y_coords_[r] = 1.0 - (global_row_begin_ + r) * h_;

    // Apply Dirichlet BCs
    apply_boundary_conditions(bc_top, bc_bottom, bc_left, bc_right);
}


// Apply BCs
void Grid::apply_boundary_conditions(const Field& bc_top,
                                     const Field& bc_bottom,
                                     const Field& bc_left,
                                     const Field& bc_right)
{
    // Top boundary: global row 0, y = 1
    // Local row index for the first owned row is always 1 (row 0 is ghost top).
    if (global_row_begin_ == 0)
    {
        const double y = y_coords_[0]; // = 1.0
        for (int j = 0; j < n_; ++j)
        {
            const double val = bc_top(x_coords_[j], y);
            (*this)(1, j) = val;
            new_(1, j) = val;
        }
    }

    // Bottom boundary: global row n-1, y = 0
    if (global_row_end_ == n_ - 1)
    {
        const double y = y_coords_[local_n_ - 1]; // = 0.0
        for (int j = 0; j < n_; ++j)
        {
            const double val = bc_bottom(x_coords_[j], y);
            (*this)(local_n_, j) = val;
            new_(local_n_, j) = val;
        }
    }

    // Left and right boundaries: columns 0 and n-1 (all owned rows)
    for (int r = 0; r < local_n_; ++r)
    {
        const double y = y_coords_[r];

        const double val_left  = bc_left (0.0, y);
        const double val_right = bc_right(1.0, y);

        (*this)(r + 1, 0) = val_left;
        new_(r + 1, 0)    = val_left;

        (*this)(r + 1, n_ - 1) = val_right;
        new_(r + 1, n_ - 1)    = val_right;
    }
}
