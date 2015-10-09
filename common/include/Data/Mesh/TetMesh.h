#ifndef TETMESH_H
#define TETMESH_H

#include <QThread>

union Data_Mesh_Tetmesh {
    struct {
        int v0,v1,v2,v3;
    };
    int data[4];
};

#ifndef USING_CUDA

    #include <vector>
    #include <SCI/Utility.h>
    #include <Data/PointData.h>
    #include <Data/MultiDimensionalDataVector.h>

    #include <Data/Mesh/PointMesh.h>
    #include <Data/Mesh/TriMesh.h>
    #include <Data/Mesh/SolidMesh.h>

    namespace Data {

        namespace Mesh {


            class TetMesh : public SolidMesh {

            public:
                TetMesh( );
                TetMesh( const char * tet_file );

                virtual bool Save( const char * ofile );

                void Clear( );

                void RecalculateTrimesh( );

                virtual void BuildSpacePartition( const PointMesh & verts );

                virtual int  GetElementCount(){ return (int)tet_mesh.size(); }

                virtual int  ContainingElement( SCI::Vex3 p, const PointMesh & verts );

                virtual void ElementIndices( int elem, std::vector<int> & ind );

                virtual float Volume( const PointMesh & verts );

                virtual void ExtractPlane( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out );

                virtual void ExtractIsotetrahedron( TetMesh & out, const PointData & pdata, float iso_val );
                virtual void ExtractIsosurface(     const PointMesh & verts_in, const PointData & pdata_in, float iso_val, int dim, TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;
                virtual void ExtractMinIsosurface(  const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;
                virtual void ExtractMeanIsosurface( const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;
                virtual void ExtractMaxIsosurface(  const PointMesh & verts_in, const PointData & pdata_in, float iso_val,          TriMesh & tris_out, PointMesh & verts_out, PointData * param_in = 0, PointData * param_out = 0 ) const;

                virtual void ExtractEdges( EdgeMesh & edges );

            public:
                std::vector<Tetrahedron> tet_mesh;

            };

            class TetMeshThread : public QThread {
                Q_OBJECT

            public:

                TetMeshThread( TetMesh * _parent );

                virtual void ExtractIsotetrahedron( TetMesh & out, const PointData & pdata, float iso_val, int start, int end );

            protected:
                virtual void run();

                TetMesh * parent;

                bool isotet;
                TetMesh * out;
                const PointData * pdata;
                float iso_val;
                int  t_start;
                int t_end;

            };
        }
    }
#endif

#endif // TETMESH_H
