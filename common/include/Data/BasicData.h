#ifndef BASICDATA_H
#define BASICDATA_H

#include <vector>

namespace Data {

    // Base class for volume holding data containers
    class BasicData {
    public:
        BasicData( int _sx, int _sy, int _sz, int _dim );

        // Dimension of each data point
        int GetDim() const ;

        // Get the size in the X,Y,Z direction
        int GetX() const ;
        int GetY() const ;
        int GetZ() const ;

        // Total number of voxels (w*h*d)
        int GetVoxelCount() const ;

        // Translate x,y,z location to voxel id
        int GetVoxelID( int x, int y, int z ) const ;

        // Get a rough estimate of the size of the data contained in the class
        virtual int GetDataSize()   = 0;

        // Various functions for setting voxel values
        virtual void SetElement( int x, int y, int z, int dim, float val )              = 0;
        virtual void SetElement( int x, int y, int z, std::vector<float> & val )        = 0;
        virtual void SetElement( int x, int y, int z, float  * val )                    = 0;
        virtual void SetElement( int x, int y, int z, double * val )                    = 0;

        virtual void SetElement( int vox_id, std::vector<float> & val )                 = 0;
        virtual void SetElement( int vox_id, int dim, float val )                       = 0;
        virtual void SetElement( int vox_id, float  * val )                             = 0;
        virtual void SetElement( int vox_id, double * val )                             = 0;

        // Various functions for getting voxel values
        virtual float              GetElement( int x, int y, int z, int dim )        const = 0;
        virtual std::vector<float> GetElement( int x, int y, int z )                 const = 0;
        virtual void               GetElement( int x, int y, int z, float  * space ) const = 0;
        virtual void               GetElement( int x, int y, int z, double * space ) const = 0;

        virtual std::vector<float> GetElement( int vox_id )                       const = 0;
        virtual float              GetElement( int vox_id, int dim )              const = 0;
        virtual void               GetElement( int vox_id, float  * space )       const = 0;
        virtual void               GetElement( int vox_id, double * space )       const = 0;

        virtual float             GetMaximumValue( )                              const = 0;
        virtual float             GetMinimumValue( )                              const = 0;

    protected:
        int     dim;
        int     sx, sy, sz;

    };



}

#endif // BASICDATA_H
