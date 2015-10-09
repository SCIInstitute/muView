#ifndef PROXYDATA_H
#define PROXYDATA_H

#include <string>

#include <Data/BasicData.h>
#include <SCI/Vex3.h>

namespace Data {

    // Generic (dense) volume data container
    class ProxyData : public BasicData {
    public:
        ProxyData(  const char * ip, unsigned short port, int _sx, int _sy, int _sz, int _dim );
        ~ProxyData();

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
        virtual float              GetElement( int x, int y, int z, int dim );
        virtual std::vector<float> GetElement( int x, int y, int z );
        virtual void               GetElement( int x, int y, int z, float  * space );
        virtual void               GetElement( int x, int y, int z, double * space );

        virtual std::vector<float> GetElement( int vox_id );
        virtual float              GetElement( int vox_id, int dim );
        virtual void               GetElement( int vox_id, float  * space );
        virtual void               GetElement( int vox_id, double * space );

        // Get a rough estimate of the size of the data contained in the class
        virtual int GetDataSize();

        virtual float             GetMaximumValue( );
        virtual float             GetMinimumValue( );

        virtual std::string       GetIP() const { return ip; }
        virtual unsigned short    GetPort() const { return port; }

        std::vector<int> conn;
        std::vector<SCI::Vex3> vert;

    protected:
        std::string    ip;
        unsigned short port;

    };
}




#endif // PROXYDATA_H
