#include <dolfin.h>
#include "IBMesh.h"
#include "NavierStokes.h"

using namespace dolfin;

// Define noslip domain
class NoslipDomain : public SubDomain
{
	bool inside(const Array<double> &x, bool on_boundary) const
	{
		return near(x[0], 0) || near(x[0], 1.0) || near(x[1], 0) || near(x[1], 1.0) || near(x[2], 0.0);
	}
};

// Define inflow domain
class InflowDomain : public SubDomain
{
	bool inside(const Array<double> &x, bool on_boundary) const
	{
		return near(x[2], 1.0);
	}
};

// Define pinpoint domain
class PinpointDomain : public SubDomain
{
	bool inside(const Array<double> &x, bool on_boundary) const
	{
		return near(x[0], 0.0) && near(x[1], 0.0) && near(x[2], 0.0);
	}
};

int main()
{
	// set time variables
	double dt = 0.01;
	double T = 10;

	// Read mesh and sub domain markers
	Point p0(0, 0, 0);
	Point p1(1, 1, 1);
	IBMesh ib_mesh({p0, p1}, {16, 16, 16});

	// Create function space
	auto W = std::make_shared<NavierStokes::FunctionSpace>(ib_mesh.mesh_ptr());

	// Define subdomains for boundary conditions
	auto noslip_domain = std::make_shared<NoslipDomain>();
	auto inflow_domain = std::make_shared<InflowDomain>();
	auto pinpoint_domain = std::make_shared<PinpointDomain>();

	// Define values for boundary conditions
	auto v_in = std::make_shared<Constant>(1.0, 0.0, 0.0);
	auto zero = std::make_shared<Constant>(0.0);
	auto zero_vector = std::make_shared<Constant>(0.0, 0.0, 0.0);

	// No-slip boundary condition for velocity
	DirichletBC bc0(W->sub(0), zero_vector, noslip_domain);

	// Inflow boundary condition for velocity
	DirichletBC bc1(W->sub(0), v_in, inflow_domain);

	// Pinpoint value for pressure
	DirichletBC bc2(W->sub(1), zero, pinpoint_domain, "pointwise");

	// Collect boundary conditions
	std::vector<const DirichletBC *> bcs = {{&bc0, &bc1, &bc2}};

	// Define coefficients
	auto f = std::make_shared<Constant>(0.0, 0.0, 0.0);
	auto k = std::make_shared<Constant>(dt);
	auto wn = Function(W);
	auto un = std::make_shared<Function>(wn[0]);
	auto pn = std::make_shared<Function>(wn[1]);
	std::cout << "tag." << std::endl;

	// Define variational problem
	NavierStokes::BilinearForm a(W, W);
	NavierStokes::LinearForm L(W);
	L.f = f;
	L.k = k;
	a.k = k;
	L.un = un;
	std::cout << "tag." << std::endl;


	// Variables to put into results
	Vector b;
	Matrix A;
	assemble(A, a);
	Function w(W);
	std::cout << "tag." << std::endl;


	// Create output files
	File ufile("results/velocity.pvd");
	File pfile("results/pressure.pvd");
	std::cout << "tag." << std::endl;

	
	// Time iteration
	for (double t = dt; t < T; t = t + dt)
	{
		// Assemble and apply boundary conditions
		assemble(b, L);
		for (std::size_t i = 0; i < bcs.size(); i++)
			bcs[i]->apply(A, b);
		
		// Compute the solution.
		solve(A, *w.vector(), b, "bicgstab", "hypre_amg");


		// Output velocity and pressure
		/// *** Error:   Unable to initialize vector of degrees of freedom for function.
		/// *** Reason:  Cannot re-initialize a non-empty vector. Consider creating a new function.
		/// *** Where:   This error was encountered inside Function.cpp.
		/// *** Process: 0
		Function u_ = w[0];
		Function p_ = w[1];
		ufile << u_;
		pfile << p_;

		// update velocity
		*un->vector() = *u_.vector();
	}
	return 0;
}
