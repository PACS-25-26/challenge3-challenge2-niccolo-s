#ifndef PROBLEM_DATA_HPP
#define PROBLEM_DATA_HPP

#include "grid.hpp"
#include <string>

/**
 * @file problem_data.hpp
 * @brief Reads problem data, parameters and boundary conditions from a JSON file.
 *
 * The JSON file specifies:
 *   - Grid and solver parameters (n, tolerance, max_iter, output file)
 *   - Forcing term as a math expression string (evaluated via exprtk)
 *   - Exact solution as a math expression string (optional, used to compute L2 error if present)
 *   - Boundary conditions (type + value expression + alpha if Robin BC type)
 *
 * Example JSON:
 * {
 *     "n":             64,
 *     "tolerance":     1e-6,
 *     "max_iter":      50000,
 *     "output":        "solution.vtk",
 *     "forcing":       "8*pi^2*sin(2*pi*x)*sin(2*pi*y)",
 *     "exact_solution": "sin(2*pi*x)*sin(2*pi*y)",
 *     "bc_top":    { "type": "dirichlet", "value": "0.0" },
 *     "bc_bottom": { "type": "dirichlet", "value": "0.0" },
 *     "bc_left":   { "type": "dirichlet", "value": "0.0" },
 *     "bc_right":  { "type": "neumann",   "value": "2*pi*cos(2*pi*y)" }
 * }
 *
 * Supported BC types: "dirichlet", "neumann", "robin"
 * Robin additionally requires an "alpha" field.
 *
 * Expression syntax (exprtk):
 *   Variables: x, y
 *   Constants: pi, e
 *   Functions: sin, cos, exp, log, sqrt, abs, ...
 */

struct ProblemData
{
    // Solver parameters
    int n;
    double tolerance;
    int max_iter;
    std::string output_file;

    // Forcing term f(x,y)
    Field forcing;

    // Exact solution (optional)
    bool has_exact = false;
    Field exact_solution;

    // Boundary conditions
    BoundaryCondition bc_top;
    BoundaryCondition bc_bottom;
    BoundaryCondition bc_left;
    BoundaryCondition bc_right;
};

/**
 * @brief Load problem data from a JSON file.
 *
 * Parses the JSON, compiles the expression strings via exprtk,
 * and returns a fully initialised ProblemData struct.
 * If the JSON contains an "exact_solution" field, has_exact is set
 * to true and exact_solution is compiled; otherwise has_exact is false.
 *
 * @param filename Path to the JSON input file.
 * @return ProblemData object.
 * @throws std::runtime_error if the file cannot be opened, the JSON is
 *         badly defined, or an expression fails to compile.
 */
ProblemData load_problem(const std::string& filename);

#endif // PROBLEM_DATA_HPP
