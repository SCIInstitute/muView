#include <Data/BasicData.h>

using namespace Data;

BasicData::BasicData( int _sx, int _sy, int _sz, int _dim ) : dim(_dim), sx(_sx), sy(_sy), sz(_sz) { }

// Functions for accessing properties of the data
int BasicData::GetDim()        const { return dim; }
int BasicData::GetX()          const { return sx;  }
int BasicData::GetY()          const { return sy;  }
int BasicData::GetZ()          const { return sz;  }
int BasicData::GetVoxelCount() const { return GetX()*GetY()*GetZ(); }

// Returns index to voxel
int BasicData::GetVoxelID( int x, int y, int z ) const {
    return z*GetX()*GetY() + y*GetX() + x;
}

