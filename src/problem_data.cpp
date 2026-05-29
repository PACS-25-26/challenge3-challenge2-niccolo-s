#include "problem_data.hpp"

#include <nlohmann/json.hpp>
#include <exprtk/exprtk.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>

using json = nlohmann::json;

// compile_expression
static Field compile_expression(const std::string& expr_str)
{
    struct ExprContext
    {
        double x = 0.0, y = 0.0;
        exprtk::symbol_table<double> symbol_table;
        exprtk::expression<double>   expression;
        exprtk::parser<double>       parser;
    };

    auto ctx = std::make_shared<ExprContext>();

    ctx->symbol_table.add_variable("x", ctx->x);
    ctx->symbol_table.add_variable("y", ctx->y);
    ctx->symbol_table.add_constants();
    ctx->expression.register_symbol_table(ctx->symbol_table);

    if (!ctx->parser.compile(expr_str, ctx->expression))
        throw std::runtime_error(
            "Expression parse error in \"" + expr_str + "\": " +
            ctx->parser.error());

    return [ctx](double x, double y) mutable -> double {
        ctx->x = x;
        ctx->y = y;
        return ctx->expression.value();
    };
}


// parse boundary conditions
static BoundaryCondition parse_bc(const json& j, const std::string& name)
{
    if (!j.contains("type"))
        throw std::runtime_error("BC \"" + name + "\" is missing \"type\"");
    if (!j.contains("value"))
        throw std::runtime_error("BC \"" + name + "\" is missing \"value\"");

    const std::string type_str = j["type"].get<std::string>();
    const std::string value_str = j["value"].get<std::string>();
    Field value = compile_expression(value_str);

    if (type_str == "dirichlet")
        return BoundaryCondition::dirichlet(std::move(value));

    if (type_str == "neumann")
        return BoundaryCondition::neumann(std::move(value));

    if (type_str == "robin")
    {
        if (!j.contains("alpha"))
            throw std::runtime_error(
                "BC \"" + name + "\" is of type robin but is missing \"alpha\"");
        return BoundaryCondition::robin(std::move(value), j["alpha"].get<double>());
    }

    throw std::runtime_error(
        "BC \"" + name + "\": unknown type \"" + type_str +
        "\". Valid types: dirichlet, neumann, robin.");
}

// Load Problem
ProblemData load_problem(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open input file: " + filename);

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(
            "JSON parse error in " + filename + ": " + e.what());
    }

    ProblemData pd;
    pd.n = j.at("n").get<int>();
    pd.tolerance = j.at("tolerance").get<double>();
    pd.max_iter = j.at("max_iter").get<int>();
    pd.output_file = j.value("output", "solution.vtk");

    if (pd.n < 2)
        throw std::runtime_error("n must be >= 2");

    // Forcing term
    pd.forcing = compile_expression(j.at("forcing").get<std::string>());

    // Exact solution (optional)
    if (j.contains("exact_solution"))
    {
        pd.has_exact      = true;
        pd.exact_solution = compile_expression(
                                j["exact_solution"].get<std::string>());
    }

    // Boundary conditions
    pd.bc_top = parse_bc(j.at("bc_top"), "bc_top");
    pd.bc_bottom = parse_bc(j.at("bc_bottom"), "bc_bottom");
    pd.bc_left = parse_bc(j.at("bc_left"), "bc_left");
    pd.bc_right = parse_bc(j.at("bc_right"), "bc_right");

    return pd;
}
