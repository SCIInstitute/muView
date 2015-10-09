#ifndef DISTANCEFIELD_H
#define DISTANCEFIELD_H

#include <QThread>

#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/TriMesh.h>
#include <Data/Mesh/SolidMesh.h>

namespace Data {

    class DistanceField {

    public:
        DistanceField( FILE *  infile );
        DistanceField( bool _min_iso, bool _max_iso, float _iso_val );
        ~DistanceField();

        void Process( Data::Mesh::PointMesh * p_mesh, Data::Mesh::SolidMesh * s_mesh, Data::PointData * p_data );

        bool Load( FILE *  infile );
        bool Save( FILE * outfile );

        void PrintInfo( );

    public:
        std::vector< float > vals;

    protected:
        Data::Mesh::PointMesh iso_points;
        Data::Mesh::TriMesh   iso_tris;

        bool  min_iso;
        bool  max_iso;
        float iso_val;

        friend class DistanceFieldSet;
    };

    class DistanceFieldSet {

    public:
        DistanceFieldSet( const char * fname );
        DistanceFieldSet();
        ~DistanceFieldSet();

        DistanceField * Insert( FILE *  infile );
        DistanceField * Insert( bool _min_iso, bool _max_iso, float _iso_val );

        void ExtractMedianIsosurface( float isoval, const Data::Mesh::SolidMesh & mesh_in, const Data::Mesh::PointMesh & points_in, Data::Mesh::TriMesh & df_tris, Data::Mesh::PointMesh & df_points, Data::PointData & df_dist );

        bool Load( const char * fname );
        bool Save( const char * fname );

    public:
        std::vector<DistanceField*> df;

    };

    class DistanceFieldThread : public QThread {

        Q_OBJECT

    public:
        DistanceFieldThread( Data::Mesh::PointMesh * p_mesh, Data::Mesh::PointMesh & iso_points, Data::Mesh::TriMesh & iso_tris, float * vals, int * elems, int elemN );
        ~DistanceFieldThread();

    protected:
        virtual void run();

        Data::Mesh::PointMesh * iso_points;
        Data::Mesh::TriMesh   * iso_tris;
        float                 * vals;
        Data::Mesh::PointMesh * p_mesh;

        //int v_start;
        //int v_end;

        int * elems;
        int elemN;


    };

}

#endif // DISTANCEFIELD_H
