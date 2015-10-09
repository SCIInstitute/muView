#ifndef EDGEMESH_H
#define EDGEMESH_H

#include <Data/Mesh/Primitives.h>
#include <Data/Mesh/ColorMesh.h>
#include <Data/Mesh/PointMesh.h>

#include <vector>

#include <SCI/Utility.h>
#include <SCI/Mat4.h>


namespace Data {

    namespace Mesh {

        class EdgeMesh : public std::vector<Edge> {

        public:
            EdgeMesh( );

            virtual Edge & GetElement( int id );
            virtual int    GetElementCount( );

            virtual void   RemoveDuplicates( );

            //void Draw( PointMesh & pmesh, ColorMesh & colors );
            void Draw( const std::vector<int> & render_order, PointMesh & pmesh, ColorMesh & colors );
            void Draw( PointMesh & pmesh, SCI::Vex4 color );

            //void SortByPainters( SCI::Mat4 proj, Data::Mesh::PointMesh & pnts, Data::Mesh::ColorMesh & cm );
            void SortByPainters(std::vector<int> &render_order, SCI::Mat4 proj, const PointMesh &pnts);


        };
    }
}



#endif // EDGEMESH_H
