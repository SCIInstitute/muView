#ifndef POINTDATA_H
#define POINTDATA_H


#include <Data/DenseMultiDimensionalData.h>

namespace Data {

    class PointData : public DenseMultiDimensionalData {

    public:
        PointData( const char * filename, bool binary );
        PointData( const PointData & data0, const PointData & data1 );
        PointData( int elem, int dim );

        std::string GetFilename( );

        inline int GetDim() const { return GetDimension(); }
        inline int GetVoxelCount() const { return GetElementCount(); }

        void Save( const char * fname );

        virtual float GetMinimumValue( int elem ) const ;
        virtual float GetMeanValue( int elem ) const ;
        virtual float GetMaximumValue( int elem ) const ;

        virtual float GetMaximumValue( ) ;
        virtual float GetMinimumValue( ) ;

    protected:
        std::string fname;

        std::vector< float > mins;
        std::vector< float > maxs;
        std::vector< float > means;

        void CalculateMinMaxMeans( );

    };
}

/*
#include <Data/VolData.h>

#include <SCI/Vex3.h>

namespace Data {

    class PointData : public VolData {

    public:
        //PointData( const char **data_files );
        PointData( const char * filename, bool binary );
        PointData( const PointData & data0, const PointData & data1 );
        PointData( int elem, int dim );

        std::string GetFilename( );

        void Save( const char * fname );

        virtual float GetMinimumValue( int elem ) const ;
        virtual float GetMeanValue( int elem ) const ;
        virtual float GetMaximumValue( int elem ) const ;

        virtual float GetMaximumValue( ) const ;
        virtual float GetMinimumValue( ) const ;

    protected:
        std::string fname;

        std::vector< float > mins;
        std::vector< float > maxs;
        std::vector< float > means;

        void CalculateMinMaxMeans( );

    };
}
*/



#endif // POINTDATA_H
