#include "mpi_utils.hpp"

// Data Exchange
void data_exchange(Grid& g, MPI_Comm comm)
{
    const int n       = g.n();
    const int local_n = g.local_n();
    const int tag     = 0;

    // We send local row 1 (first owned row) to rank_above.
    // We receive into local row 0 (ghost top) from rank_above.
    if (g.rank_above() != -1)
    {
        MPI_Sendrecv(
            g.row_ptr(1),        // send buffer: first owned row
            n, MPI_DOUBLE,
            g.rank_above(), tag, // destination
            g.row_ptr(0),        // recv buffer: ghost top
            n, MPI_DOUBLE,
            g.rank_above(), tag, // source
            comm, MPI_STATUS_IGNORE
        );
    }

    // We send local row local_n (last owned row) to rank_below.
    // We receive into local row local_n+1 (ghost bottom) from rank_below.
    if (g.rank_below() != -1)
    {
        MPI_Sendrecv(
            g.row_ptr(local_n),      // send buffer: last owned row
            n, MPI_DOUBLE,
            g.rank_below(), tag,     // destination
            g.row_ptr(local_n + 1),  // recv buffer: ghost bottom
            n, MPI_DOUBLE,
            g.rank_below(), tag,     // source
            comm, MPI_STATUS_IGNORE
        );
    }
}


// Global iteration L2 difference computation
double check_convergence(const Grid& g, double local_sum, MPI_Comm comm)
{
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(g.h() * global_sum);
}