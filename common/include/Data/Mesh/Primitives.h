#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <SCI/Vex2.h>
#include <SCI/Vex3.h>
#include <SCI/Vex4.h>
#include <SCI/BoundingBox.h>

namespace Data {
    class PointData;
    class MultiDimensionalDataVector;

    namespace Mesh {

        class PointMesh;
        class TriMesh;
        class EdgeMesh;

        class Edge {
        public:
            Edge( int _v0 = -1, int _v1 = -1 );

            int GetSmaller() const ;
            int GetLarger()  const ;

            bool operator <  ( const Edge & right ) const ;
            bool operator == ( const Edge & right ) const ;

        public:
            int v0, v1;
        };


        class Triangle {
        public:
            Triangle( FILE * file, bool binary );
            Triangle( int _v0 = -1, int _v1 = -1, int _v2 = -1 );

            bool operator <  ( const Triangle right ) const ;
            bool operator != ( const Triangle right ) const ;
            bool operator == ( const Triangle right ) const ;

            void Write( FILE * outfile );

            SCI::Vex3        GetNormal( const PointMesh &pmesh );
            float            GetArea( const PointMesh &pmesh );
            float            GetDistance( SCI::Vex3 pnt, const PointMesh &pmesh, bool quick_estimate = false );
            SCI::BoundingBox GetBoundingBox( const PointMesh & pmesh );

            Edge GetEdge(int i);
            void ExtractIsoline( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const std::vector<float> & pdata_in, float iso_val );

            void Flip();

            bool isDegenerate();

            union{
                struct {
                    int v0,v1,v2;
                };
                int data[3];
            };
        };


        class Quadrilateral {
        public:
            Quadrilateral( FILE * file, bool binary );
            Quadrilateral( int _v0 = -1, int _v1 = -1, int _v2 = -1, int _v3 = -1 );

            bool operator <  ( const Quadrilateral right ) const ;
            bool operator != ( const Quadrilateral right ) const ;

            Triangle GetTriangle(int i);
            Edge     GetEdge(int i);

            union{
                struct {
                    int v0,v1,v2,v3;
                };
                int data[4];
            };


        };



        class Tetrahedron {
        public:
            Tetrahedron( FILE * file, bool binary );
            Tetrahedron( const char * buf );
            Tetrahedron( int _v0, int _v1, int _v2, int _v3 );

            bool  isValid();
            bool  isPointContained( SCI::Vex3 p, const PointMesh & pmesh );

            float GetVolume( const PointMesh & pmesh );
            void  GetBoundingBox( SCI::Vex3 & bbmin, SCI::Vex3 & bbmax, const PointMesh & pmesh );

            void  ExtractSurface( TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, float val0, float val1, float val2, float val3, float iso_val, const PointData * param_in = 0, PointData * param_out = 0 );
            void  ExtractSurface( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out );

            union{
                struct {
                    int v0,v1,v2,v3;
                };
                int data[4];
            };

        };

        class Hexahedron {
        public:
            Hexahedron( FILE * infile, bool binary );
            Hexahedron( const char * buf );
            Hexahedron( int _v0, int _v1, int _v2, int _v3, int _v4, int _v5, int _v6, int _v7 );
            Hexahedron( );

            bool  isValid();

            void  Write( FILE * outfile );

            float             GetVolume( const PointMesh & pmesh );
            SCI::BoundingBox  GetBoundingBox( const PointMesh & pmesh );
            Tetrahedron       GetTetrahedron( int i ) const ;
            Quadrilateral     GetQuadrilateral( int i );
            Edge              GetEdge( int i );

            union{
                struct {
                    int v0,v1,v2,v3,v4,v5,v6,v7;
                };
                int data[8];
            };

        };
    }

}

#endif // PRIMITIVES_H
