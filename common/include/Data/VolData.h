#ifndef VOLDATA_H
#define VOLDATA_H

/*
#include <Data/BasicData.h>

namespace Data {

    #pragma deprecated(VolData)

    // Generic (dense) volume data container
    class VolData : public BasicData {
    public:
        VolData(  int _sx, int _sy, int _sz, int _dim, float *_vol = 0 );
        ~VolData();

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

        virtual float             GetMaximumValue( ) const ;
        virtual float             GetMinimumValue( ) const ;

    protected:
        void __resize(  int _sx, int _sy, int _sz, int _dim );
        float * vol;
        bool    readonly;
    };
}

*/

#endif // VOLDATA_H
