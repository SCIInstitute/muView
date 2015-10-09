#ifndef MAPPEDDATA_H
#define MAPPEDDATA_H

/*
#include <map>

#include <Data/BasicData.h>

namespace Data {

    // Compact (sparse) data storage for subsets of data elements
    class MappedData : public BasicData
    {
    public:
        MappedData( int _sx, int _sy, int _sz, int _dim );
        ~MappedData();

        // Various functions for setting voxel values
        virtual void SetElement( int x, int y, int z, int dim, float val );
        virtual void SetElement( int x, int y, int z, std::vector<float> & val );
        virtual void SetElement( int x, int y, int z, float  * val );
        virtual void SetElement( int x, int y, int z, double * val );

        virtual void SetElement( int vox_id, std::vector<float> & val );
        virtual void SetElement( int vox_id, int dim, float val );
        virtual void SetElement( int vox_id, float  * val );
        virtual void SetElement( int vox_id, double * val );

        // Various functions for getting voxel values
        virtual float              GetElement( int x, int y, int z, int dim ) const ;
        virtual std::vector<float> GetElement( int x, int y, int z ) const ;
        virtual void               GetElement( int x, int y, int z, float  * space ) const ;
        virtual void               GetElement( int x, int y, int z, double * space ) const ;

        virtual std::vector<float> GetElement( int vox_id ) const ;
        virtual float              GetElement( int vox_id, int dim ) const ;
        virtual void               GetElement( int vox_id, float  * space ) const ;
        virtual void               GetElement( int vox_id, double * space ) const ;

        // Get a rough estimate of the size of the data contained in the class
        virtual int GetDataSize();

        virtual float             GetMaximumValue( );
        virtual float             GetMinimumValue( );


    protected:
        std::map<int,float*>  data_location;
        std::vector< float* > alloc_list;
        int                   alloc_space_avail;
        float *               alloc_space_pointer;

        float * GetDataLocation( int voxID, bool alloc = false );
        float * GetDataLocation( int voxID ) const ;
    };

}

*/

#endif // MAPPEDDATA_H
