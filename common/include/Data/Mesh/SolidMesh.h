#ifndef SOLIDMESH_H
#define SOLIDMESH_H


#include <vector>
#include <SCI/Utility.h>
#include <Data/Mesh/TriMesh.h>
#include <Data/Mesh/PointMesh.h>
#include <Data/PointData.h>
#include <Data/MultiDimensionalDataVector.h>

#include <Data/Mesh/KDTree.h>

namespace Data {

    namespace Mesh {

        class TetMesh;
        class HexMesh;

        class SolidMesh {

        public:
            SolidMesh( ){ }

            virtual bool Save( const char * ofile ) = 0;

            virtual void BuildSpacePartition( const PointMesh & verts ) = 0;

            virtual int  ContainingElement( SCI::Vex3 p, const PointMesh & verts ) = 0;
            virtual void ElementIndices( int elem, std::vector<int> & ind ) = 0;

            virtual float Volume( const PointMesh & verts ) = 0;

            virtual int  GetElementCount() = 0;

            virtual void ExtractEdges( EdgeMesh & edges ) = 0;

            virtual void ExtractPlane( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out ) = 0;

            virtual void ExtractIsotetrahedron( TetMesh & out, const PointData & pdata, float iso_val );
            virtual void ExtractIsohexahedron(  HexMesh & out, const PointData & pdata, float iso_val );

            /*
            virtual void ExtractIsosurface(     TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata, int dim, float iso_val ) = 0;
            virtual void ExtractMinIsosurface(  TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata,          float iso_val ) = 0;
            virtual void ExtractMeanIsosurface( TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata,          float iso_val ) = 0;
            virtual void ExtractMaxIsosurface(  TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata,          float iso_val ) = 0;
            */

            virtual void ExtractIsosurface(     const PointMesh & verts_in, const PointData & pdata_in, float iso_val, int dim, TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const = 0;
            virtual void ExtractMinIsosurface(  const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const = 0;
            virtual void ExtractMeanIsosurface( const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const = 0;
            virtual void ExtractMaxIsosurface(  const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const = 0;


        public:

            TriMesh tri_mesh;

            KDTree kdtree;

        };
    }
}



#endif // SOLIDMESH_H
