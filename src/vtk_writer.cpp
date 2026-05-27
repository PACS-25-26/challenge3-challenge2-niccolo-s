#include "vtk_writer.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

void write_vtk(const Grid& g, const std::string& filename, MPI_Comm comm)
{
    const int n = g.n();
    const double h = g.h();
    const int local_n = g.local_n();
    const int rank = g.rank();
    const int size = g.size();
    const int tag = 0;

    // Each rank saves its owned rows into a vector, ghost rows are skipped (local indices 0 and local_n+1).
    std::vector<double> local_buf(local_n * n);
    for (int r = 0; r < local_n; ++r)
        for (int c = 0; c < n; ++c)
            local_buf[r * n + c] = g(r + 1, c); // r+1 skips ghost top

    // All non-zero ranks send their buffer to rank 0 and return
    if (rank != 0)
    {
        MPI_Send(local_buf.data(), local_n * n, MPI_DOUBLE, 0, tag, comm);
        return;
    }

    // Assemble the full solution on rank 0
    std::vector<double> global_buf(n * n, 0.0);

    // Copy rank 0's own rows
    for (int r = 0; r < local_n; ++r)
        for (int c = 0; c < n; ++c)
            global_buf[r * n + c] = local_buf[r * n + c];

    // Receive from other ranks in order
    int row_offset = local_n;
    for (int src = 1; src < size; ++src)
    {
        int src_begin, src_end;
        Grid::compute_row_range(n, src, size, src_begin, src_end);
        int src_local_n = src_end - src_begin + 1;

        MPI_Recv(global_buf.data() + row_offset * n,
                 src_local_n * n, MPI_DOUBLE,
                 src, tag, comm, MPI_STATUS_IGNORE);

        row_offset += src_local_n;
    }

    // Write the VTK file on rank 0
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("write_vtk: cannot open file " + filename);

    // VTK header
    out << "# vtk DataFile Version 3.0\n";
    out << "Laplace equation solution\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << n << " " << n << " 1\n";
    out << "ORIGIN 0.0 0.0 0.0\n";
    out << "SPACING " << h << " " << h << " 1.0\n";
    out << "POINT_DATA " << n * n << "\n";
    out << "SCALARS solution double 1\n";
    out << "LOOKUP_TABLE default\n";

    // Write solution values.
    // Since our convention is y = 1 - global_row * h (see grid.cpp), row 0 corresponds
    // to y=1 (top) and row n-1 to y=0 (bottom).
    // VTK STRUCTURED_POINTS expects y=0 first, so we write in reverse row order.
    for (int r = n - 1; r >= 0; --r)
        for (int c = 0; c < n; ++c)
            out << global_buf[r * n + c] << "\n";
}
