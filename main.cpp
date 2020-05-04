

#include <dolfin.h>
#include "TetPoisson.h"
#include "HexPoisson.h"
#include "IBMesh.h"
#include "IBInterpolation.h"
#include "LocalPolynomial.h"

using namespace dolfin;

class xyz : public Expression
{
public:
	xyz() : Expression(3) {}
	void eval(Array<double> &values, const Array<double> &x) const
	{
		values[0] = x[0];
		values[1] = x[1];
        values[2] = x[2];
	}
};

int main()
{
	size_t nx;
	std::cin >> nx;
	auto g = std::make_shared<xyz>();

	Point p0(0, 0, 0);
	Point p1(1, 1, 1);
	IBMesh ib_mesh({p0, p1}, {nx, nx, nx});
	auto U = std::make_shared<HexPoisson::FunctionSpace>(ib_mesh.mesh_ptr());
	Function u(U);
	u.interpolate(*g);

    auto sphere = std::make_shared<Mesh>("./sphere.xml.gz");
	auto V = std::make_shared<TetPoisson::FunctionSpace>(sphere);
	Function v(V);
    
    std::cout<<"interpolation"<<std::endl;
	DeltaInterplation di(ib_mesh);
	di.fluid_to_solid(u,v);

	File file("xyz.pvd");
	file << v;

/*
auto sum = 0.0;
	auto aa = v.function_space()->tabulate_dof_coordinates();
	for (size_t i = 0; i < aa.size()/9; i++)
	{
		std::cout<<"error1: "
		         << aa[i*9] -v.vector()->getitem(i*3) << std::endl;
	sum+=abs(aa[i*9] -v.vector()->getitem(i*3));
	}
	for (size_t i = 0; i < aa.size()/9; i++)
	{
		std::cout<<"error2: "
		         << aa[i*9+1] -v.vector()->getitem(i*3+1) << std::endl;
	sum+=abs(aa[i*9+1] -v.vector()->getitem(i*3+1));
	}
	for (size_t i = 0; i < aa.size()/9; i++)
	{
		std::cout<<"error3: "
		         << aa[i*9+2] -v.vector()->getitem(i*3+2) << std::endl;
	sum+=abs(aa[i*9+2] -v.vector()->getitem(i*3+2));
	}
	std::cout<<sum<<std::endl;
*/
}
