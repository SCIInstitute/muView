#include <Data/ProxyData.h>
#include <float.h>

using namespace Data;

// Constructor
ProxyData::ProxyData(  const char *_ip, unsigned short _port, int _sx, int _sy, int _sz, int _dim ) : BasicData( _sx, _sy, _sz, _dim ), ip(_ip), port(_port) { }

// Destructor
ProxyData::~ProxyData(){ }

float ProxyData::GetMaximumValue( ){
    return 0;
}

float ProxyData::GetMinimumValue( ){
    return 0;
}

// Get a rough estimate of the size of the data contained in the class
int ProxyData::GetDataSize(){
    return sizeof(float)*sx*sy*sz*dim;
}

// Wrapper functions to Set Values
void ProxyData::SetElement( int , int , int , int , float ){       }
void ProxyData::SetElement( int , int , int , std::vector<float> & ){ }
void ProxyData::SetElement( int , int , int , float  * ){             }
void ProxyData::SetElement( int , int , int , double * ){             }

// Set an element from a vector
void ProxyData::SetElement( int vox_id, std::vector<float> & val ){ }

// Set a single dimension of a voxel
void ProxyData::SetElement( int vox_id, int dim, float val ){ }

// Set an element from a float array
void ProxyData::SetElement( int vox_id, float  * val ){ }

// Set an element from a double array
void ProxyData::SetElement( int vox_id, double * val ){ }

// Wrapper functions to Read Values
float              ProxyData::GetElement( int x, int y, int z, int dim ){        return GetElement( GetVoxelID(x,y,z), dim );    }
std::vector<float> ProxyData::GetElement( int x, int y, int z ){                 return GetElement( GetVoxelID(x,y,z) );         }
void               ProxyData::GetElement( int x, int y, int z, float * space ){  return GetElement( GetVoxelID(x,y,z), space );  }
void               ProxyData::GetElement( int x, int y, int z, double * space ){ return GetElement( GetVoxelID(x,y,z), space );  }

// Get a single dimension from an element
float ProxyData::GetElement( int vox_id, int dim ){
    return 0;
}

// Get an element as a vector
std::vector<float> ProxyData::GetElement( int vox_id ){
    std::vector<float> ret;
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        ret.push_back( 0 );
    }
    return ret;
}

// Copy an element into a float array
void ProxyData::GetElement( int vox_id, float * space ){
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        space[c_dim] = 0;
    }
}

// Copy an element into a double array
void ProxyData::GetElement( int vox_id, double * space ){
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        space[c_dim] = 0;
    }
}


