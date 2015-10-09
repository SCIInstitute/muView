#ifndef QUADMESH_H
#define QUADMESH_H

#include <Data/Mesh/Primitives.h>
#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/TriMesh.h>

#include <SCI/Utility.h>
#include <SCI/Mat4.h>
#include <vector>


namespace Data {

    namespace Mesh {


        class QuadMesh {

        public:
            QuadMesh();

            void RemoveSharedFaces( );
            void ConvertToTrimesh( TriMesh & tri_mesh );
            //void SortByPainters( SCI::Mat4 proj, Data::Mesh::PointMesh & pnts );

            virtual void ExtractEdges( EdgeMesh & edges );

        public:
            std::vector<Quadrilateral>    quads;

        };
    }
}




#endif // QUADMESH_H
