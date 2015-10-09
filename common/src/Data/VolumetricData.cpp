#include <Data/VolumetricData.h>

using namespace Data;

VolumetricData::VolumetricData(  int _sx, int _sy, int _sz, int _dim ) : DenseMultiDimensionalData(_sx*_sy*_sz,_dim), sx(_sx), sy(_sy), sz(_sz){

}

// Get the size in the X,Y,Z direction
int VolumetricData::GetX() const { return sx; }
int VolumetricData::GetY() const { return sy; }
int VolumetricData::GetZ() const { return sz; }

// Total number of voxels (w*h*d)
int VolumetricData::GetVoxelCount() const { return GetElementCount(); }

// Translate x,y,z location to voxel id
int VolumetricData::GetVoxelID( int x, int y, int z ) const {
    return z*GetX()*GetY() + y*GetX() + x;
}

// Various functions for setting voxel values
void VolumetricData::SetElement( int x, int y, int z, int dim, float val ){
    DenseMultiDimensionalData::SetElement( GetVoxelID(x,y,z), dim, val );
}

void VolumetricData::SetElement( int x, int y, int z, std::vector<float> & val ){
    DenseMultiDimensionalData::SetElement( GetVoxelID(x,y,z), val );
}

void VolumetricData::SetElement( int x, int y, int z, float  * val ){
    DenseMultiDimensionalData::SetElement( GetVoxelID(x,y,z), val );
}

void VolumetricData::SetElement( int x, int y, int z, double * val ){
    DenseMultiDimensionalData::SetElement( GetVoxelID(x,y,z), val );
}

// Various functions for getting voxel values
float VolumetricData::GetElement( int x, int y, int z, int dim ) const {
    return DenseMultiDimensionalData::GetElement( GetVoxelID(x,y,z), dim );
}

std::vector<float> VolumetricData::GetElement( int x, int y, int z ) const  {
    return DenseMultiDimensionalData::GetElement( GetVoxelID(x,y,z) );
}

void VolumetricData::GetElement( int x, int y, int z, float  * space ) const  {
    DenseMultiDimensionalData::GetElement( GetVoxelID(x,y,z), space );
}

void VolumetricData::GetElement( int x, int y, int z, double * space ) const  {
    DenseMultiDimensionalData::GetElement( GetVoxelID(x,y,z), space );
}
