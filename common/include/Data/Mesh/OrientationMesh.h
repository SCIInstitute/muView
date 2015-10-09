#ifndef ORIENTATIONMESH_H
#define ORIENTATIONMESH_H

#include <Data/Mesh/Primitives.h>

#include <SCI/Utility.h>
#include <SCI/Mat4.h>
#include <vector>

#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/KDTree.h>
#include <Data/Mesh/EdgeMesh.h>
#include <Data/Mesh/TriMesh.h>

namespace Data {

    namespace Mesh {

        class OrientationMesh : public std::vector<SCI::Vex3> {

        public:
            OrientationMesh();

            void Calculate( TriMesh & tmesh, PointMesh & pmesh );
            void Smooth( const TriMesh & tmesh, int iterations = 1 );
            void ZeroFlatRegions( TriMesh & tmesh, PointMesh & pmesh );

        };
    }
}

#endif // ORIENTATIONMESH_H
