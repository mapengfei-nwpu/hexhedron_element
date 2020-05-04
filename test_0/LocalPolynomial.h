#ifndef __LOCALPOLYNOMIAL_H
#define __LOCALPOLYNOMIAL_H
#include <map>
#include <vector>
#include <dolfin.h>
#include <Eigen/Dense>
#include <Eigen/LU>
namespace dolfin
{

class LocalPolynomial
{
public:
    size_t value_size = 3;
    size_t poly_degree = 27;
    Function function;

    std::map<size_t, std::vector<std::vector<double>>> polynomial_coefficients;

    std::vector<double> evaluate_polynomial_items(Point point);

    double accumulate(std::vector<double> weights, std::vector<double> parameters);

    std::vector<double> eval(const Point &point, size_t cell_index);

    std::vector<std::vector<double>> calculate_coefficients(size_t cell_index);
};
} // namespace dolfin
#endif