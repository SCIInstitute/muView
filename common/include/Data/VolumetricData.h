#ifndef VOLUMETRICDATA_H
#define VOLUMETRICDATA_H

#include <Data/DenseMultiDimensionalData.h>

namespace Data {

    // Base class for volume holding data containers
    class VolumetricData : public DenseMultiDimensionalData {
    public:
        VolumetricData(  int _sx, int _sy, int _sz, int _dim );//, float *_vol = 0 );
        ~VolumetricData();

        // Get the size in the X,Y,Z direction
        int GetX() const ;
        int GetY() const ;
        int GetZ() const ;

        // Total number of voxels (w*h*d)
        int GetVoxelCount() const ;

        // Translate x,y,z location to voxel id
        int GetVoxelID( int x, int y, int z ) const ;

        // Various functions for setting voxel values
        virtual void SetElement( int x, int y, int z, int dim, float val );
        virtual void SetElement( int x, int y, int z, std::vector<float> & val );
        virtual void SetElement( int x, int y, int z, float  * val );
        virtual void SetElement( int x, int y, int z, double * val );

        // Various functions for getting voxel values
        virtual float              GetElement( int x, int y, int z, int dim ) const ;
        virtual std::vector<float> GetElement( int x, int y, int z ) const ;
        virtual void               GetElement( int x, int y, int z, float  * space ) const ;
        virtual void               GetElement( int x, int y, int z, double * space ) const ;

    protected:
        int sx, sy, sz;
        //float * vol;
        //bool    readonly;
    };



}




#endif // VOLUMETRICDATA_H
