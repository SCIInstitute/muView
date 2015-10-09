#ifndef HEXMESH_H
#define HEXMESH_H



#include <vector>
#include <SCI/Utility.h>
#include <Data/PointData.h>

#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/TriMesh.h>
#include <Data/Mesh/SolidMesh.h>
#include <Data/Mesh/QuadMesh.h>
#include <Data/Mesh/TetMesh.h>

namespace Data {

    namespace Mesh {


        class HexMesh : public SolidMesh {

        public:
            HexMesh( );
            HexMesh( const char * hex_file );

            void Clear();

            virtual bool Save( const char * ofile );

            void RecalculateTrimesh( );

            virtual float Volume( const PointMesh & verts );

            virtual int  GetElementCount(){ return (int)hex_mesh.size(); }

            virtual void BuildSpacePartition( const PointMesh & verts );

            virtual int  ContainingElement( SCI::Vex3 p, const PointMesh & verts );

            virtual void ExtractEdges( EdgeMesh & edges );

            virtual void ExtractPlane( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out ) ;

            virtual void ElementIndices( int elem, std::vector<int> & ind );

            virtual void ExtractIsohexahedron(  HexMesh & out, const PointData & pdata, float iso_val );

            /*
            virtual void ExtractIsosurface(     TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata, int dim, float iso_val );
            virtual void ExtractMinIsosurface(  TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata,          float iso_val );
            virtual void ExtractMeanIsosurface( TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata,          float iso_val );
            virtual void ExtractMaxIsosurface(  TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata,          float iso_val );
            */

            virtual void ExtractIsosurface(     const PointMesh & verts_in, const PointData & pdata_in, float iso_val, int dim, TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;
            virtual void ExtractMinIsosurface(  const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;
            virtual void ExtractMeanIsosurface( const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;
            virtual void ExtractMaxIsosurface(  const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;


        protected:

            void isosurface_fill_values(  int type, const Hexahedron & hex, const PointData & pdata,    float               vals_out[2][2][2], int dim = -1 ) const;
            void isosurface_fill_verts(             const Hexahedron & hex, const PointMesh & verts_in, SCI::Vex3          verts_out[2][2][2]               ) const;
            void isosurface_fill_params(            const Hexahedron & hex, const PointData * verts_in, std::vector<float> verts_out[2][2][2]               ) const;
            bool isosurface_need_compute( float vals[2][2][2], float iso_val ) const;
            void isosurface_process( int type, int dim, TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata, float iso_val, PointData *param_in, PointData *param_out ) const;

            //void ExtractSurface( TriMesh & tris, PointMesh & verts, const PointData & pdata, Hexahedron & tet, float val0, float val1, float val2, float val3 , float iso_val );

        public:
            std::vector<Hexahedron> hex_mesh;

            QuadMesh quad_mesh;


        };
    }
}


#endif // HEXMESH_H
