#include "jacobi_solver.hpp"
#include "mpi_utils.hpp"

#include <cmath>

// Jacobi method
SolverResult jacobi_solve(Grid& g,
                           const Field& f,
                           const BoundaryCondition& bc_top,
                           const BoundaryCondition& bc_bottom,
                           const BoundaryCondition& bc_left,
                           const BoundaryCondition& bc_right,
                           double tol,
                           int max_iter,
                           MPI_Comm comm)
{
    const int n = g.n();
    const double h = g.h();
    const double h2 = h * h;
    const int local_n = g.local_n();

    // Column range: include boundary columns only if not Dirichlet
    // (Neumann/Robin boundary columns are updated by the Jacobi loop)
    const int c_start = (bc_left.type  == BCType::Dirichlet) ? 1 : 0;
    const int c_stop  = (bc_right.type == BCType::Dirichlet) ? n - 2 : n - 1;

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
            const int  global_r = g.global_row_begin() + r - 1;
            const bool is_top = (global_r == 0);
            const bool is_bottom = (global_r == n - 1);

            // Skip Dirichlet boundary rows entirely
            if (is_top && bc_top.type == BCType::Dirichlet) continue;
            if (is_bottom && bc_bottom.type == BCType::Dirichlet) continue;

            const double y = g.y_coord(r - 1);

            for (int c = c_start; c <= c_stop; ++c)
            {
                const double x = g.x_coord(c);
                const bool is_left = (c == 0);
                const bool is_right = (c == n - 1);

                // ── Neighbour values ──────────────────────────────────────────
                // For interior neighbours: use the actual adjacent value.
                // For Neumann/Robin boundaries: ghost node method.
                //
                // Ghost node derivation (left boundary as example):
                //   BC: du/dn + alpha*u = g  with n = -x  =>  -du/dx + alpha*u = g
                //   Centred diff: (u_{i,1} - u_{i,-1}) / (2h) = alpha*u_{i,0} - g
                //   => u_{i,-1} = u_{i,1} - 2h*(alpha*u_{i,0} - g)
                //              = u_{i,1} + 2h*(g - alpha*u_{i,0})
                //
                // Substituting into the stencil, u_{i,-1} + u_{i,1} becomes:
                //   2*u_{i,1} + 2h*(g - alpha*u_{i,0})
                //
                // The alpha*u_{i,0} term moves to the left-hand side,
                // increasing the denominator by 2h*alpha.
                // For Neumann (alpha=0): u_{ghost} = u_{adjacent} + 2h*g,
                // denominator unchanged (stays 4).

                // North boundary (y=1)
                double u_north;
                if (is_top)
                {
                    const double bc_val = bc_top.value(x, y);
                    // outward normal at top points in +y: du/dn = +du/dy
                    // centered diff: (u_ghost - u_{r+1,c}) / (2h) = alpha*u - g
                    // => u_ghost = u_{r+1,c} + 2h*(g - alpha*u_{r,c})
                    u_north = g(r + 1, c) + 2.0 * h * (bc_val - bc_top.alpha * g(r, c));
                }
                else
                    u_north = g(r - 1, c);

                // South boundary (y=0)
                double u_south;
                if (is_bottom)
                {
                    const double bc_val = bc_bottom.value(x, y);
                    // outward normal at bottom points in -y: du/dn = -du/dy
                    // => u_ghost = u_{r-1,c} + 2h*(g - alpha*u_{r,c})
                    u_south = g(r - 1, c) + 2.0 * h * (bc_val - bc_bottom.alpha * g(r, c));
                }
                else
                    u_south = g(r + 1, c);

                // West boundary (x=0)
                double u_west;
                if (is_left)
                {
                    const double bc_val = bc_left.value(x, y);
                    // outward normal at left points in -x: du/dn = -du/dx
                    // => u_ghost = u_{r,c+1} + 2h*(g - alpha*u_{r,c})
                    u_west = g(r, c + 1) + 2.0 * h * (bc_val - bc_left.alpha * g(r, c));
                }
                else
                    u_west = g(r, c - 1);

                // East boundary (x=1)
                double u_east;
                if (is_right)
                {
                    const double bc_val = bc_right.value(x, y);
                    // outward normal at right points in +x: du/dn = +du/dx
                    // => u_ghost = u_{r,c-1} + 2h*(g - alpha*u_{r,c})
                    u_east = g(r, c - 1) + 2.0 * h * (bc_val - bc_right.alpha * g(r, c));
                }
                else
                    u_east = g(r, c + 1);

                // Denominator
                // Base value is 4. Each Robin boundary on this node contributes
                // +2h*alpha to the denominator (from moving alpha*u to the LHS).
                double denom = 4.0;
                if (is_top && bc_top.type == BCType::Robin)
                    denom += 2.0 * h * bc_top.alpha;

                if (is_bottom && bc_bottom.type == BCType::Robin)
                    denom += 2.0 * h * bc_bottom.alpha;

                if (is_left && bc_left.type == BCType::Robin)
                    denom += 2.0 * h * bc_left.alpha;

                if (is_right && bc_right.type == BCType::Robin)
                    denom += 2.0 * h * bc_right.alpha;

                // Jacobi update
                const double new_val =
                    (u_north + u_south + u_west + u_east + h2 * f(x, y)) / denom;

                const double diff = new_val - g(r, c);
                local_sum += diff * diff;

                g.new_(r, c) = new_val;
            }
        }

        // Global convergence check
        const double error = check_convergence(g, local_sum, comm);

        // Swap buffers
        g.swap_buffers();

        // Re-enforce Dirichlet BCs
        // Neumann/Robin nodes are updated by the loop above.
        // Dirichlet nodes must be reset since they must never change.
        g.apply_boundary_conditions(bc_top, bc_bottom, bc_left, bc_right);

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