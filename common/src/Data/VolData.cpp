#include <Data/VolData.h>
#include <float.h>

/*
using namespace Data;

// Constructor
VolData::VolData(  int _sx, int _sy, int _sz, int _dim, float * _vol ) : BasicData( _sx, _sy, _sz, _dim ) {
    if(_vol == 0 ){
        readonly = false;
        vol = new float[sx*sy*sz*dim];
    }
    else{
        readonly = true;
        vol = _vol;
    }
}

void VolData::__resize( int _sx, int _sy, int _sz, int _dim ) {
    sx = _sx;
    sy = _sy;
    sz = _sz;
    dim = _dim;
    if(!readonly){
        delete [] vol;
        vol = new float[sx*sy*sz*dim];
    }
}

float VolData::GetMaximumValue( ) const {
    float mval = -FLT_MAX;
    for(int i = 0; i < sx*sy*sz*dim; i++){
        mval = std::max( mval, vol[i] );
    }
    return mval;
}

float VolData::GetMinimumValue( ) const {
    float mval = FLT_MAX;
    for(int i = 0; i < sx*sy*sz*dim; i++){
        mval = std::min( mval, vol[i] );
    }
    return mval;
}

// Destructor
VolData::~VolData(){
    if(!readonly && vol != 0)
        delete vol;
}

// Get a rough estimate of the size of the data contained in the class
int VolData::GetDataSize(){
    return sizeof(float)*sx*sy*sz*dim;
}

// Wrapper functions to Set Values
void VolData::SetElement( int x, int y, int z, int dim, float val ){       SetElement( GetVoxelID(x,y,z), dim, val );  }
void VolData::SetElement( int x, int y, int z, std::vector<float> & val ){ SetElement( GetVoxelID(x,y,z), val );    }
void VolData::SetElement( int x, int y, int z, float  * val ){             SetElement( GetVoxelID(x,y,z), val );   }
void VolData::SetElement( int x, int y, int z, double * val ){             SetElement( GetVoxelID(x,y,z), val );   }

// Set an element from a vector
void VolData::SetElement( int vox_id, std::vector<float> & val ){
    if(readonly) return;
    for(int c_dim = 0; c_dim < GetDim() && c_dim < (int)val.size(); c_dim++){
        vol[ c_dim*GetVoxelCount() + vox_id] = val[c_dim];
    }
}

// Set a single dimension of a voxel
void VolData::SetElement( int vox_id, int dim, float val ){
    if(readonly) return;
    vol[ dim*GetVoxelCount() + vox_id ] = val;
}

// Set an element from a float array
void VolData::SetElement( int vox_id, float  * val ){
    if(readonly) return;
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        vol[ c_dim*GetVoxelCount() + vox_id] = val[c_dim];
    }
}

// Set an element from a double array
void VolData::SetElement( int vox_id, double * val ){
    if(readonly) return;
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        vol[ c_dim*GetVoxelCount() + vox_id] = (float)val[c_dim];
    }
}

// Wrapper functions to Read Values
float              VolData::GetElement( int x, int y, int z, int dim ) const {        return GetElement( GetVoxelID(x,y,z), dim );    }
std::vector<float> VolData::GetElement( int x, int y, int z ) const {                 return GetElement( GetVoxelID(x,y,z) );         }
void               VolData::GetElement( int x, int y, int z, float * space ) const {  return GetElement( GetVoxelID(x,y,z), space );  }
void               VolData::GetElement( int x, int y, int z, double * space ) const { return GetElement( GetVoxelID(x,y,z), space );  }

// Get a single dimension from an element
float VolData::GetElement( int vox_id, int dim ) const {
    return vol[ dim*GetVoxelCount() + vox_id ];
}

// Get an element as a vector
std::vector<float> VolData::GetElement( int vox_id ) const {
    std::vector<float> ret;
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        ret.push_back( vol[ c_dim*GetVoxelCount() + vox_id] );
    }
    return ret;
}

// Copy an element into a float array
void VolData::GetElement( int vox_id, float * space ) const {
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        space[c_dim] = vol[ c_dim*GetVoxelCount() + vox_id];
    }
}

// Copy an element into a double array
void VolData::GetElement( int vox_id, double * space ) const {
    for(int c_dim = 0; c_dim < GetDim(); c_dim++){
        space[c_dim] = vol[ c_dim*GetVoxelCount() + vox_id];
    }
}

*/
