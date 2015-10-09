#ifndef COLORMESH_H
#define COLORMESH_H

#include <SCI/Vex3.h>
#include <SCI/Vex4.h>
#include <SCI/Utility.h>
#include <SCI/Colormap.h>

#include <vector>


class KMeansClustering;

namespace Data {
    class MultiDimensionalData;

    namespace Mesh {

        class OrientationMesh;
        class PointMesh;
        class SolidMesh;
        class TriMesh;

        class ColorMesh : public std::vector<unsigned int> {

        public:
            ColorMesh();

            virtual void clear();

                  unsigned int & GetColor( int i );
            const unsigned int & GetColor( int i ) const ;
                  unsigned int & DefaultColor( );
            const unsigned int & DefaultColor( ) const ;

            bool isColorPerVertex()  const ;
            bool isColorPerElement() const ;

            void SetColorPerVertex();
            void SetColorPerElement();

            void Fill(const PointMesh & pmesh , SCI::Vex4 color);
            void Fill(TriMesh & smesh , SCI::Vex4 color);
            void SetByDataDimension( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap , int dim);
            void SetByDataMin( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap );
            void SetByDataMax( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap );
            void SetByDataMean( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap );
            void SetByDataStdev( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap );
            void SetByDataClusterID( const KMeansClustering & km, SCI::CatagoricalColormap & cmap );
            void SetFromOrientation(OrientationMesh &ort_mesh, SCI::Vex3 vd, float power);
            void SetByDataIsoRange(const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap, SCI::Vex4 mincol, SCI::Vex4 maxcol , float isoval);

        public:
            unsigned int def_color;

        protected:
            bool cpv, cpe;

            void __SetByValueAndColormap__( std::vector< float > val, SCI::SequentialColormap & cmap );
        };
    }
}

#endif // COLORMESH_H
