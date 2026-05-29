#include "grid.hpp"
#include "jacobi_solver.hpp"
#include "vtk_writer.hpp"
#include "problem_data.hpp"

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// L2 error norm against the exact solution
static double compute_l2_error(const Grid& g, const Field& exact, MPI_Comm comm)
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
    if (argc < 2)
    {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <problem.json>\n";
        MPI_Finalize();
        return 1;
    }

    // Load problem data
    ProblemData pd;
    try {
        pd = load_problem(argv[1]);
    } catch (const std::exception& e) {
        if (rank == 0)
            std::cerr << "ERROR loading problem: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        std::cout << "=== Jacobi solver ===\n"
                  << "  grid size : " << pd.n << " x " << pd.n << "\n"
                  << "  tolerance : " << pd.tolerance << "\n"
                  << "  max iter  : " << pd.max_iter << "\n"
                  << "  MPI ranks : " << size << "\n"
                  << "  output    : " << pd.output_file << "\n"
                  << "  exact sol : " << (pd.has_exact ? "yes" : "no") << "\n\n";
    }

    // Build grid
    Grid g(pd.n, rank, size,
           pd.bc_top, pd.bc_bottom, pd.bc_left, pd.bc_right);

    // Solve
    const double t_start = MPI_Wtime();

    SolverResult result = jacobi_solve(g, pd.forcing,
                                       pd.bc_top,    pd.bc_bottom,
                                       pd.bc_left,   pd.bc_right,
                                       pd.tolerance, pd.max_iter);


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

   // L2 error (only if exact solution is provided in JSON)
    if (pd.has_exact)
    {
        const double l2_err = compute_l2_error(g, pd.exact_solution,
                                               MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "  L2 error: approximate vs exact solution: "
                      << l2_err << "\n\n";
    }

    // Write VTK
    write_vtk(g, pd.output_file);

    if (rank == 0)
        std::cout << "Solution written to " << pd.output_file << "\n";

    MPI_Finalize();
    return 0;
}