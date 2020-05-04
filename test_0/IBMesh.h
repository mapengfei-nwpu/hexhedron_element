#ifndef _IBMESH_H_
#define _IBMESH_H_
#include <iostream>
#include <dolfin.h>
namespace dolfin
{

class IBMesh
{
public:
	// Mesh characteristics
	double x0, x1, y0, y1, z0, z1;
	std::size_t nx, ny, nz;
	std::size_t top_dim;
	std::size_t mpi_rank;

	// The map of global index to hash index for cells.
	std::vector<std::array<std::size_t, 2>> global_map;
	Mesh mesh;

	IBMesh(std::array<Point, 2> points, std::vector<std::size_t> dims,
		   CellType::Type cell_type = CellType::Type::hexahedron);

	std::vector<std::size_t> get_adjacents(Point point, int bandwidth = 3);

	// global index to local index
	std::array<std::size_t, 2> map(std::size_t i);

	// TODO : will this function create another copy of mesh?
	std::shared_ptr<Mesh> mesh_ptr();

	// return the length of every side of the mesh.
	std::vector<double> cell_length();

	// given a global cell index i, global_map return a tuple (a,b),
	// where "a" is the local index and "b" is the mpi processor of
	// this cell.
	void index_mesh();

	// Every point has a unique cell index.
	// It is consistent with the generation of the mesh.
	std::size_t hash(Point point);
};
} // namespace dolfin
#endif