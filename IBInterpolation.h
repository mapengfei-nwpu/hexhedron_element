#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <numeric>
#include <ctime>
#include "IBMesh.h"
#include "HexahedronEvaluation.h"
using namespace dolfin;

template <typename T>
std::vector<T> my_mpi_gather(std::vector<T> local)
{
	auto mpi_size = MPI::size(MPI_COMM_WORLD);

	/// collect local values on every process
	std::vector<std::vector<T>> mpi_collect(mpi_size);
	MPI::all_gather(MPI_COMM_WORLD, local, mpi_collect);
	std::vector<T> global;

	/// unwrap mpi_collect
	for (size_t i = 0; i < mpi_collect.size(); i++)
	{
		/// TODO : I failed to do the following step.
		/// global.insert(global.end(), mpi_collect[i].begin(), mpi_collect.end());
		for (size_t j = 0; j < mpi_collect[i].size(); j++)
		{
			global.push_back(mpi_collect[i][j]);
		}
	}
	return global;
}

std::vector<std::array<double, 3>> get_global_dof_coordinates(const Function &function)
{
	auto mesh = function.function_space()->mesh();
	auto local_dof_coordinates = function.function_space()->tabulate_dof_coordinates();
	auto dof_coordinates_long = my_mpi_gather(local_dof_coordinates);
	auto value_size = function.value_size();

	std::size_t DIM = 3;
	dolfin_assert(DIM == mesh->topology.dim());

	std::vector<std::array<double, 3>> dof_coordinates(dof_coordinates_long.size() / DIM / value_size);
	for (size_t i = 0; i < dof_coordinates_long.size(); i += DIM * value_size)
	{
		for (size_t j = 0; j < DIM; j++)
		{
			dof_coordinates[i / DIM / value_size][j] = dof_coordinates_long[i + j];
		}
	}

	return dof_coordinates;
}

void get_gauss_rule(
	const Function &function,
	std::vector<double> &coordinates,
	std::vector<double> &values,
	std::vector<double> &weights)
{
	auto order = 6;
	auto mesh = function.function_space()->mesh();
	auto dim = mesh->topology().dim();
	auto value_size = function.value_size();

	// Construct Gauss quadrature rule
	SimplexQuadrature gq(dim, order);

	for (CellIterator cell(*mesh); !cell.end(); ++cell)
	{
		// Create ufc_cell associated with dolfin cell.
		ufc::cell ufc_cell;
		cell->get_cell_data(ufc_cell);

		// Compute quadrature rule for the cell.
		auto qr = gq.compute_quadrature_rule(*cell);
		dolfin_assert(qr.second.size() == qr.first.size() / 2);
		for (size_t i = 0; i < qr.second.size(); i++)
		{
			Array<double> v(dim);
			Array<double> x(dim, qr.first.data());

			/// Call evaluate function
			function.eval(v, x, *cell, ufc_cell);

			/// push back what we get.
			weights.push_back(qr.second[i]);
			for (size_t d = 0; d < dim; d++)
			{
				coordinates.push_back(x[d]);
			}
			for (size_t d = 0; d < value_size; d++)
			{
				values.push_back(v[d]);
			}
		}
	}
	values = my_mpi_gather(values);
	weights = my_mpi_gather(weights);
	coordinates = my_mpi_gather(coordinates);
}

class DeltaInterplation
{
public:
	IBMesh &ib_mesh;
	std::vector<double> side_lengths;

	DeltaInterplation(IBMesh &ib_mesh) : ib_mesh(ib_mesh)
	{
		side_lengths = ib_mesh.cell_length();
	}

	void fluid_to_solid(Function &fluid, Function &solid)
	{
		auto mesh = fluid.function_space()->mesh();
		auto global_solid_size = solid.function_space()->dim();
		auto mpi_rank = MPI::rank(fluid.function_space()->mesh()->mpi_comm());

		/// I wrote it and it is for hexahedron evaluation.
		/// it is only for second order element with any value size.
		HexaheronEvaluation hexa_eval;
		hexa_eval.evaluate_basis_coefficients();

		/// calculate global dof coordinates and dofs of solid.
		std::vector<std::array<double, 3>> solid_dof_coordinates = get_global_dof_coordinates(solid);
		std::vector<double> solid_local_values(global_solid_size);

		for (std::size_t i = 0; i < solid_dof_coordinates.size(); ++i)
		{
			Point point(
				solid_dof_coordinates[i][0], 
				solid_dof_coordinates[i][1], 
				solid_dof_coordinates[i][2]);
			auto hash_index = ib_mesh.hash(point);
			auto mpi_rank_and_local_index = ib_mesh.global_map[hash_index];
			if (mpi_rank_and_local_index[0] == mpi_rank)
			{
				/// Find the dolfin cell where point reside.
				auto local_index = mpi_rank_and_local_index[1];
				Cell cell(*mesh, local_index);

				auto values = hexa_eval.eval(point, cell, fluid);

				solid_local_values[i * 3] = values[0];
				solid_local_values[i * 3 + 1] = values[1];
				solid_local_values[i * 3 + 2] = values[2];

				std::cout<<values[0]<<","
				<<values[1]<<","
				<<values[2]<<std::endl;
				std::cout<<point<<std::endl;
			}
		}

		std::vector<double> solid_values(global_solid_size);
		std::vector<std::vector<double>> mpi_collect(dolfin::MPI::size(mesh->mpi_comm()));
		MPI::all_gather(mesh->mpi_comm(), solid_local_values, mpi_collect);

		for (size_t i = 0; i < solid_values.size(); i++)
		{
			for (size_t j = 0; j < mpi_collect.size(); j++)
			{
				solid_values[i] += mpi_collect[j][i];
			}
		}
		auto local_size = solid.vector()->local_size();
		auto offset = solid.vector()->local_range().first;
		std::vector<double> local_values(solid_values.begin() + offset, solid_values.begin() + offset + local_size);
		solid.vector()->set_local(local_values);
		solid.vector()->apply("insert");
	}

	/// Assign the solid displacement with the velocity of fluid.
	void solid_to_fluid(Function &fluid, Function &solid)
	{
		std::clock_t time_start = std::clock();
		/// calculate global dof coordinates and dofs of solid.
		std::vector<double> solid_dof_coordinates;
		std::vector<double> solid_values;
		std::vector<double> weights;
		get_gauss_rule(solid, solid_dof_coordinates, solid_values, weights);
		std::clock_t time_end = std::clock();
		std::cout << "gauss quadrature spend: " << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << std::endl;

		/// interpolates solid values into fluid mesh.
		/// the returned local_fluid_values is the dofs of fluid function.
		/// dofs at shared points should be accumulated.
		auto fluid_values = solid_to_fluid_raw(fluid, solid_values, solid_dof_coordinates, weights);
		time_end = std::clock();
		std::cout << "local interpolation spend: " << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << std::endl;

		auto local_size = fluid.vector()->local_size();
		auto offset = fluid.vector()->local_range().first;
		std::vector<double> local_values(fluid_values.begin() + offset, fluid_values.begin() + offset + local_size);
		fluid.vector()->set_local(local_values);
		fluid.vector()->apply("insert");
		time_end = std::clock();
		std::cout << "set local values spend: " << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << std::endl;
	}

	std::vector<double> solid_to_fluid_raw(
		Function &fluid,
		std::vector<double> &solid_values,
		std::vector<double> &solid_coordinates,
		std::vector<double> &weights)
	{
		/// smart shortcut
		auto rank = MPI::rank(fluid.function_space()->mesh()->mpi_comm());
		auto mesh = fluid.function_space()->mesh();		// pointer to a mesh
		auto dofmap = fluid.function_space()->dofmap(); // pointer to a dofmap
		auto hmax = mesh->hmax();

		/// get the element of function space
		auto element = fluid.function_space()->element();
		auto value_size = fluid.value_size();
		auto global_fluid_size = fluid.function_space()->dim();

		/// Get local to global dofmap
		std::vector<size_t> local_to_global;
		dofmap->tabulate_local_to_global_dofs(local_to_global);

		/// initial local fluid values.
		std::vector<double> local_fluid_values(global_fluid_size);

		/// iterate every gauss point on solid.
		for (size_t i = 0; i < solid_values.size() / value_size; i++)
		{

			/// get indices of adjacent cells on fluid mesh.
			Point solid_point(solid_coordinates[3 * i], solid_coordinates[3 * i + 1], solid_coordinates[3 * i + 2]);
			auto adjacents = ib_mesh.get_adjacents(solid_point);

			/// iterate adjacent cells and collect element nodes in these cells.
			/// it has nothing to do with cell type.
			std::map<size_t, double> indices_to_delta;
			for (size_t j = 0; j < adjacents.size(); j++)
			{
				/// step 1 : get coordinates of cell dofs
				Cell cell(*mesh, adjacents[j]);
				std::vector<double> coordinate_dofs;
				cell.get_coordinate_dofs(coordinate_dofs);
				boost::multi_array<double, 2> coordinates;
				element->tabulate_dof_coordinates(coordinates, coordinate_dofs, cell);

				/// step 2 : get the dof map
				auto cell_dofmap = dofmap->cell_dofs(cell.index());

				/// iterate node coordinates of the cell.
				for (size_t k = 0; k < cell_dofmap.size() / value_size; k++)
				{
					Point cell_point(coordinates[k][0], coordinates[k][1], coordinates[k][2]);
					double param = delta(solid_point, cell_point);
					if (cell_dofmap[k] < fluid.vector()->local_size() && param > 0.0)
					{
						indices_to_delta[cell_dofmap[k]] = param;
					}
				}
			}

			for (auto it = indices_to_delta.begin(); it != indices_to_delta.end(); it++)
			{
				for (size_t l = 0; l < value_size; l++)
				{
					local_fluid_values[local_to_global[it->first + l]] += solid_values[i * value_size + l] * it->second * weights[i];
				}
			}
		}
		/// this cost about
		std::clock_t time_start = std::clock();
		std::vector<double> fluid_values(global_fluid_size);
		std::vector<std::vector<double>> mpi_collect(dolfin::MPI::size(mesh->mpi_comm()));
		dolfin::MPI::all_gather(mesh->mpi_comm(), local_fluid_values, mpi_collect);
		for (size_t i = 0; i < fluid_values.size(); i++)
		{
			for (size_t j = 0; j < mpi_collect.size(); j++)
			{
				fluid_values[i] += mpi_collect[j][i];
			}
		}
		std::clock_t time_end = std::clock();
		std::cout << "distribute: " << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << std::endl;

		return fluid_values;
	}

	/////////////////////////////////////////////
	//  thses methods must not be modified!!  ///
	/////////////////////////////////////////////
	double phi(double r)
	{
		r = fabs(r);
		if (r > 2)
			return 0;
		else
			return 0.25 * (1 + cos(FENICS_PI * r * 0.5));
	}
	double delta(Point p0, Point p1)
	{
		double ret = 1.0;
		for (unsigned i = 0; i < 3; ++i)
		{
			double dx = p0.coordinates()[i] - p1.coordinates()[i];
			ret *= 1. / side_lengths[i] * phi(dx / side_lengths[i]);
		}
		return ret;
	}
	/////////////////////////////////////////////
	//  thses methods must not be modified!!  ///
	/////////////////////////////////////////////
};
