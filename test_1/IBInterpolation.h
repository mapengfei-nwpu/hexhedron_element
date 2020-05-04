#ifndef _IBINTERPOLATION_H_
#define _IBINTERPOLATION_H_
#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <numeric>
#include <ctime>
#include "IBMesh.h"
#include "LocalPolynomial.h"
namespace dolfin
{
/// the usage of DeltaInterpolation:
/// DeltaInterpolation di(ib_mesh);
/// di.fluid_to_soild(fluid,solid);
/// di.solid_to_fluid(fluid,solid);

/// the method for interpolation can be modified:
/// bandwidth is the bandwidth,
/// delta is the delta,
/// they can be modified.
class DeltaInterplation
{
public:
	DeltaInterplation(IBMesh &ib_mesh) : ib_mesh(ib_mesh)
	{
		side_lengths = ib_mesh.cell_length();
	}

	void fluid_to_solid(Function &fluid, Function &solid);

	void solid_to_fluid(Function &fluid, Function &solid);

private:
	IBMesh &ib_mesh;
	int bandwidth = 3;
	std::vector<double> side_lengths;

	/// mid step for solid to fluid interpolation.
	std::vector<double> solid_to_fluid_raw(
		Function &fluid,
		std::vector<double> &solid_values,
		std::vector<double> &solid_coordinates,
		std::vector<double> &weights);

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
} // namespace dolfin
#endif