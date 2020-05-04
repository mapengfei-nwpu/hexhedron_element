#include "LocalPolynomial.h"
using namespace dolfin;
std::vector<double> LocalPolynomial::evaluate_polynomial_items(Point point)
{
    double x = point.x();
    double y = point.y();
    double z = point.z();

    std::vector<double> polynomial_items(poly_degree);
    polynomial_items[0] = 1.0;
    polynomial_items[1] = z;
    polynomial_items[2] = z * z;
    polynomial_items[3] = y;
    polynomial_items[4] = y * z;
    polynomial_items[5] = y * z * z;
    polynomial_items[6] = y * y;
    polynomial_items[7] = y * y * z;
    polynomial_items[8] = y * y * z * z;
    polynomial_items[9] = x;
    polynomial_items[10] = x * z;
    polynomial_items[11] = x * z * z;
    polynomial_items[12] = x * y;
    polynomial_items[13] = x * y * z;
    polynomial_items[14] = x * y * z * z;
    polynomial_items[15] = x * y * y;
    polynomial_items[16] = x * y * y * z;
    polynomial_items[17] = x * y * y * z * z;
    polynomial_items[18] = x * x;
    polynomial_items[19] = x * x * z;
    polynomial_items[20] = x * x * z * z;
    polynomial_items[21] = x * x * y;
    polynomial_items[22] = x * x * y * z;
    polynomial_items[23] = x * x * y * z * z;
    polynomial_items[24] = x * x * y * y;
    polynomial_items[25] = x * x * y * y * z;
    polynomial_items[26] = x * x * y * y * z * z;
    return polynomial_items;
}

double LocalPolynomial::accumulate(std::vector<double> weights, std::vector<double> parameters)
{
    double sum = 0.0;
    /// weights.size() == parameters.size()
    for (size_t i = 0; i < weights.size(); i++)
    {
        sum += weights[i] * parameters[i];
    }
    return sum;
}

std::vector<std::vector<double>> LocalPolynomial::calculate_coefficients(size_t cell_index)
{
    /// dofs
    auto mesh = function.function_space()->mesh();
    Cell cell(*mesh, cell_index);
    const GenericDofMap &dofmap = *(function.function_space()->dofmap());
    auto cell_dofmap = dofmap.cell_dofs(cell.index());
    std::vector<double> dofs(poly_degree * value_size);
    function.vector()->get_local(dofs.data(), cell_dofmap.size(), cell_dofmap.data());

    /// coordinates
    boost::multi_array<double, 2> coordinates;
    std::vector<double> coordinate_dofs;
    cell.get_coordinate_dofs(coordinate_dofs);
    const FiniteElement &element = *function.function_space()->element();
    element.tabulate_dof_coordinates(coordinates, coordinate_dofs, cell);

    /// inverse and multipy the right side term.
    Eigen::MatrixXd matrix(poly_degree, poly_degree);
    for (size_t i = 0; i < poly_degree; i++)
    {
        Point point(
            coordinates[i][0],
            coordinates[i][1],
            coordinates[i][2]);
        auto items = evaluate_polynomial_items(point);
        for (size_t j = 0; j < poly_degree; j++)
        {
            matrix(i, j) = items[j];
        }
    }

    auto solver = matrix.lu();
    std::vector<std::vector<double>> coefficients;

    for (size_t i = 0; i < value_size; i++)
    {
        std::vector<double> coefficient(poly_degree);
        Eigen::VectorXd b(poly_degree);
        for (size_t j = 0; j < poly_degree; j++)
        {
            b(j) = dofs[i * poly_degree + j];
        }
        auto result = solver.solve(b);
        for (size_t j = 0; j < poly_degree; j++)
        {
            coefficient[j] = result(j);
        }
        coefficients.push_back(coefficient);
    }
    return coefficients;
}
std::vector<double> LocalPolynomial::eval(const Point &point, size_t cell_index)
{
    /// To evaluate a function at a point, we need to reconstruct a polynomial
    /// function on the local cell for which polynomial coefficients should be
    /// calculated. It is very expensive to calculate and store coefficients for
    /// every cell. So, we just calculate and store what we want.

    /// if polynomial coefficients for current cell haven't been calculated,
    /// find(local_index) will return end(). In this situation, we need to
    /// calculate it and add it to the map(polynomial_coefficients)

    if (polynomial_coefficients.find(cell_index) == polynomial_coefficients.end())
    {
        polynomial_coefficients[cell_index] = calculate_coefficients(cell_index);
    }

    auto coefficients = polynomial_coefficients[cell_index];
    auto items = evaluate_polynomial_items(point);

    std::vector<double> values(value_size);
    for (size_t i = 0; i < value_size; i++)
    {
        values[i] = accumulate(coefficients[i], items);
    }
    return values;
}