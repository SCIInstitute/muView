#ifndef POINTMESH_H
#define POINTMESH_H

#include <vector>
#include <map>

#include <SCI/Vex3.h>
#include <SCI/Vex4.h>
#include <SCI/BoundingBox.h>

#include <Data/Mesh/ColorMesh.h>

namespace Data {
    namespace Mesh {
        class PointMesh
        {
        public:
            PointMesh();
            PointMesh( const char * fname, bool binary, bool compressed );

            void Save( const char * fname, bool compress );


            void Clear();

            void RecalculateBoundingBox();

            //void FillColors( unsigned int col );

            SCI::Vex3 GetVertex( int i ) const ;

            void Draw( ColorMesh & colors );
            void Draw( SCI::Vex4 override_color );

            void FuseCoindidentVertices( std::map<int,int> & idx_map, int range_min = 0, int range_max = INT_MAX, float eps = 1.0e-6 );

        public:

            //SCI::Vex3 bbmin,bbmax;
            SCI::BoundingBox bb;

            std::vector<SCI::Vex3>    points;
            //std::vector<unsigned int> colors;


        };
    }
}

#endif // POINTMESH_H
