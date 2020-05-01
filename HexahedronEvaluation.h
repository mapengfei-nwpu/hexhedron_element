#include <dolfin.h>
#include <vector>
#include <array>

using namespace dolfin;
class HexaheronEvaluation
{
public:
    /// 27 vertices.
    /// 27x27 parameters.
    std::vector<double> reference_coordinates = {
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5,
        0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5,
        0.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.5,
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5,
        1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
        1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5,
        0.5, 0.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.5,
        0.5, 1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5,
        0.5, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5};
    std::vector<std::vector<double>> coefficients;

    std::array<double, 27> evaluate_polynomial_items(Point point)
    {
        double x = point.x();
        double y = point.y();
        double z = point.z();

        std::array<double, 27> polynomial_items;
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

    void evaluate_basis_coefficients()
    {
        Eigen::MatrixXd matrix(27, 27);
        for (size_t i = 0; i < reference_coordinates.size() / 3; i++)
        {
            Point point(
                reference_coordinates[3 * i],
                reference_coordinates[3 * i + 1],
                reference_coordinates[3 * i + 2]);
            auto polynomial_items = evaluate_polynomial_items(point);
            for (size_t j = 0; j < polynomial_items.size(); j++)
            {
                matrix(j, i) = polynomial_items[j];
            }
        }
        auto inverse = matrix.inverse();
        for (size_t i = 0; i < 27; i++)
        {
            std::vector<double> coefficient;
            for (size_t j = 0; j < 27; j++)
            {
                coefficient.push_back(inverse(i,j));
            }
            coefficients.push_back(coefficient);
        }
    }

    /// transform local point to reference point
    Point transform_to_reference_point(const Cell &cell, const Point &point)
    {
        /// calculate coordinate_dofs on cell.
        std::vector<double> coordinate_dofs;
        cell.get_coordinate_dofs(coordinate_dofs);

        /// coordinate_dof is corresponding the following
        /// vertices on reference cell.
        /// {0.0, 0.0, 0.0},
        /// {0.0, 0.0, 1.0},
        /// {0.0, 1.0, 0.0},
        /// {0.0, 1.0, 1.0},
        /// {1.0, 0.0, 0.0},
        /// {1.0, 0.0, 1.0},
        /// {1.0, 1.0, 0.0},
        /// {1.0, 1.0, 1.0},

        /// Because I am using rectangular cells, the first and the last vertices are needed.
        Point pa(coordinate_dofs[0], coordinate_dofs[1], coordinate_dofs[2]);
        Point pb(coordinate_dofs[21], coordinate_dofs[22], coordinate_dofs[23]);
        Point reference_point(
            (point.x() - pa.x()) / (pb.x() - pa.x()),
            (point.y() - pa.y()) / (pb.y() - pa.y()),
            (point.z() - pa.z()) / (pb.z() - pa.z()));
        return reference_point;
    }

    std::vector<double> eval(
        const Point &point,
        const Cell &cell,
        const Function &function)
    {
        auto value_size = function.value_size();
        std::vector<double> dofs(value_size * 27);
        std::vector<double> values(value_size);
        {
            // Get dofmap for cell
            const GenericDofMap &dofmap = *(function.function_space()->dofmap());
            auto cell_dofmap = dofmap.cell_dofs(cell.index());

            // Pick values from vector(s)
            function.vector()->get_local(dofs.data(), cell_dofmap.size(), cell_dofmap.data());
        }
        auto reference_point = transform_to_reference_point(cell, point);
        auto polynomial_items = evaluate_polynomial_items(reference_point);

        /// TODO : compare value_size with dofs.size(). They should be the same.

        /// Evaluate the values.
        /// i : ith value item.
        /// j : jth basis function.
        /// k : kth polynomial item.
        for (size_t i = 0; i < value_size; i++)
        {
            for (size_t j = 0; j < coefficients.size(); j++)
            {
                double basis_value = 0.0;
                for (size_t k = 0; k < coefficients.size(); k++)
                {
                    basis_value += polynomial_items[k] * coefficients[j][k];
                }
                values[i] += dofs[j + i*27] *basis_value;
            }
        }
        return values;
    }
};