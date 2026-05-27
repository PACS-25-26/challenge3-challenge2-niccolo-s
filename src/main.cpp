#include "grid.hpp"
#include "jacobi_solver.hpp"
#include "vtk_writer.hpp"

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// Forcing term and exact solution
static double forcing(double x, double y)
{
    return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
}

static double exact(double x, double y)
{
    return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
}

// L2 error norm against the exact solution
static double compute_l2_error(const Grid& g, MPI_Comm comm)
{
    const int    n       = g.n();
    const double h       = g.h();
    const int    local_n = g.local_n();

    double local_sum = 0.0;

    for (int r = 0; r < local_n; ++r)
    {
        const int global_r = g.global_row_begin() + r;

        // skip boundary rows
        if (global_r == 0 || global_r == n - 1)
            continue;

        const double y = g.y_coord(r);

        for (int c = 1; c < n - 1; ++c)
        {
            const double x   = g.x_coord(c);
            const double diff = g(r + 1, c) - exact(x, y); // r+1: skip ghost top
            local_sum += diff * diff;
        }
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(h * global_sum);
}

// main
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check arguments
    if (argc < 4)
    {
        MPI_Finalize();
        return 1;
    }

    const int    n        = std::stoi(argv[1]);
    const double tol      = std::stod(argv[2]);
    const int    max_iter = std::stoi(argv[3]);
    const std::string vtk_file = (argc >= 5) ? argv[4] : "solution.vtk";

    if (rank == 0)
    {
        std::cout << "=== Jacobi solver ===\n"
                  << "  grid size : " << n << " x " << n << "\n"
                  << "  tolerance : " << tol << "\n"
                  << "  max iter  : " << max_iter << "\n"
                  << "  MPI ranks : " << size << "\n"
                  << "  output file name (default: solution.vtk) : " << vtk_file << "\n\n";
    }

    // Homogeneous Dirichlet BCs
    auto zero = [](double, double){ return 0.0; };

    // Build grid
    Grid g(n, rank, size, zero, zero, zero, zero);

    // Solve
    const double t_start = MPI_Wtime();

    SolverResult result = jacobi(g, forcing,
                                zero, zero, zero, zero,
                                tol, max_iter);

    const double t_end = MPI_Wtime();

    // Print results
    if (rank == 0)
    {
        std::cout << "--- Solver results ---\n"
                  << "  converged : " << (result.converged ? "yes" : "no") << "\n"
                  << "  iterations: " << result.iterations << "\n"
                  << "  final error (increment norm): " << result.error << "\n"
                  << "  computing time : " << (t_end - t_start) << " s\n\n";
    }

    // L2 error against exact solution
    const double l2_err = compute_l2_error(g, MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "  L2 error: approximate vs exact solution: " << l2_err << "\n\n";

    // Write VTK file
    write_vtk(g, vtk_file);

    if (rank == 0)
        std::cout << "Solution written to " << vtk_file << "\n";

    MPI_Finalize();
    return 0;
}