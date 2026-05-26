#include "jacobi_solver.hpp"
#include "mpi_utils.hpp"

#include <cmath>

// Jacobi method
SolverResult jacobi(Grid& g,
                    const Field& f,
                    const Field& bc_top,
                    const Field& bc_bottom,
                    const Field& bc_left,
                    const Field& bc_right,
                    double tol,
                    int max_iter,
                    MPI_Comm comm)
{
    const int n = g.n();
    const double h = g.h();
    const double h2 = h * h;
    const int local_n = g.local_n();

    SolverResult result{0, 0.0, false};

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Data exchange
        data_exchange(g, comm);

        // Jacobi update + local error sum
        double local_sum = 0.0;

        #pragma omp parallel for reduction(+:local_sum)
        for (int r = 1; r <= local_n; ++r)
        {
            const int    global_r = g.global_row_begin() + r - 1;
            if (global_r == 0 || global_r == n - 1)
                continue;

            const double y = g.y_coord(r - 1);

            for (int c = 1; c < n - 1; ++c)
            {
                const double x = g.x_coord(c);

                const double new_val =
                    0.25 * (g(r - 1, c) +
                            g(r + 1, c) +
                            g(r, c - 1) +
                            g(r, c + 1) +
                            h2 * f(x, y));

                const double diff = new_val - g(r, c);
                local_sum += diff * diff;

                g.new_(r, c) = new_val;
            }
        }

        // Global convergence check
        const double error = check_convergence(g, local_sum, comm);

        // Swap buffers
        g.swap_buffers();

        result.iterations = iter + 1;
        result.error = error;

        if (error < tol)
        {
            result.converged = true;
            break;
        }
    }

    return result;
}