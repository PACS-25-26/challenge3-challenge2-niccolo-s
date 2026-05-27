#ifndef VTK_WRITER_HPP
#define VTK_WRITER_HPP

#include "grid.hpp"
#include <mpi.h>
#include <string>

/**
 * @file vtk_writer.hpp
 * @brief Export the solution to a VTK file.
 *
 * All non-zero ranks send their local solution strip to rank 0 via MPI_Send.
 * Rank 0 collects them sequentially and writes a single VTK file.
 */

/**
 * @brief Collects the solution from all ranks and write a VTK file.
 *
 * Each rank sends its local_n owned rows (excluding ghost rows) to rank 0.
 * Rank 0 assembles the full n*n solution and writes it to `filename`.
 *
 * @param g         Grid (holds the local solution in U).
 * @param filename  Output file path. Only used by rank 0.
 * @param comm      MPI communicator.
 */
void write_vtk(const Grid& g,
               const std::string& filename,
               MPI_Comm comm = MPI_COMM_WORLD);

#endif // VTK_WRITER_HPP
