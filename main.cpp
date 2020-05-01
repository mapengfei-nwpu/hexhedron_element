#include <random>

#include <dolfin.h>
#include "HexPoisson.h"
#include "IBMesh.h"
#include "HexahedronEvaluation.h"

using namespace dolfin;

class xyz : public Expression
{
public:
	xyz() : Expression(3) {}
	void eval(Array<double> &values, const Array<double> &x) const
	{
		values[0] = x[0] * x[0];
		values[1] = x[1] * x[1];
		values[2] = x[2] * x[2];
	}
};

int main()
{
	size_t nx;
	std::random_device rd;
	std::uniform_real_distribution<double> uu(0, 1);
	std::cin >> nx;
	auto g = std::make_shared<xyz>();

	Point p0(0, 0, 0);
	Point p1(1, 1, 1);
	IBMesh ib_mesh({p0, p1}, {nx, nx, nx});
	auto mesh = ib_mesh.mesh_ptr();
	auto U = std::make_shared<HexPoisson::FunctionSpace>(ib_mesh.mesh_ptr());
	Function u(U);
	u.interpolate(*g);

	HexaheronEvaluation hexa_eval;
	hexa_eval.evaluate_basis_coefficients();
	std::cout<<"tag"<<std::endl;
	while (true)
	{

		Point p2(uu(rd), uu(rd), uu(rd));
		auto global_index = ib_mesh.hash(p2);
		auto local_index = ib_mesh.global_map[global_index][1];
		Cell cell(*mesh, local_index);

		auto values = hexa_eval.eval(p2, cell, u);

		std::cout << sqrt(values[0]) << "," << sqrt(values[1]) << "," << sqrt(values[2]) << std::endl;
		std::cout << p2 << std::endl;

		std::cout<<"cell index: "<<cell.index()<<std::endl;
		std::cout<<"hash:"<<ib_mesh.hash(p2)<<std::endl;
		std::cout<<"local_index:"<<local_index<<std::endl;


	}
}
