#include <Data/MappedData.h>

/*
#include <malloc.h>
#include <float.h>
#include <iostream>
#include <stdio.h>

using namespace Data;

#define MAPPEDDATA_ALLOC_SIZE 2048

// Constructor
MappedData::MappedData(int _sx, int _sy, int _sz, int _dim) : BasicData( _sx, _sy, _sz, _dim ){
    // create initial data space
    alloc_list.push_back( new float[GetDim()*MAPPEDDATA_ALLOC_SIZE] );
    alloc_space_avail = MAPPEDDATA_ALLOC_SIZE;
    alloc_space_pointer = alloc_list.back();
}

// Destruction
MappedData::~MappedData(){
    for(int i = 0; i < (int)alloc_list.size(); i++){
        delete [] alloc_list[i];
    }
}

float MappedData::GetMaximumValue( ){
    std::cout << "ERROR: Not yet implemented" << std::endl; fflush(stdout);
    return 0;
}

float MappedData::GetMinimumValue( ){
    std::cout << "ERROR: Not yet implemented" << std::endl; fflush(stdout);
    return 0;
}


// Returns the approximate size of this object
int MappedData::GetDataSize(){
    return (int)alloc_list.size() * sizeof(float) * GetDim() * MAPPEDDATA_ALLOC_SIZE;
}

// Get the location in memory for a specific voxel ID
float * MappedData::GetDataLocation( int voxID, bool alloc ){
    // check if location exists
    if( data_location.find( voxID ) == data_location.end() ){
        if(!alloc) return 0;

        // allocate space, if none exists
        if( alloc_space_avail <= 0 ){
            alloc_list.push_back( new float[GetDim()*MAPPEDDATA_ALLOC_SIZE] );
            alloc_space_avail = MAPPEDDATA_ALLOC_SIZE;
            alloc_space_pointer = alloc_list.back();
        }

        // set location to next available space
        data_location[voxID] = alloc_space_pointer;
        alloc_space_pointer += GetDim();
        alloc_space_avail--;
    }
    // return the location
    return data_location[ voxID ];
}

// Get the location in memory for a specific voxel ID
float * MappedData::GetDataLocation( int voxID ) const {
    // check if location exists
    if( data_location.find( voxID ) == data_location.end() ){
        return 0;
    }
    // return the location
    return data_location.find( voxID )->second;
}

// Wrappers for real SetElement functions
void MappedData::SetElement( int x, int y, int z, float  * val ){             SetElement( GetVoxelID(x,y,z), val );       }
void MappedData::SetElement( int x, int y, int z, int dim, float val ){       SetElement( GetVoxelID(x,y,z), dim, val );  }
void MappedData::SetElement( int x, int y, int z, std::vector<float> & val ){ SetElement( GetVoxelID(x,y,z), &(val[0]) ); }
void MappedData::SetElement( int x, int y, int z, double * val ){             SetElement( GetVoxelID(x,y,z), val );       }
void MappedData::SetElement( int vox_id, std::vector<float> & val ){          SetElement( vox_id, &(val[0]) );            }

// Set element for a single point, single dimension
void MappedData::SetElement( int vox_id, int dim, float val ){
    float * loc = GetDataLocation(vox_id,true);
    loc[dim] = val;
}

// Set element for a single point
void MappedData::SetElement( int vox_id, float  * val ){
    float * loc = GetDataLocation(vox_id,true);
    for(int i = 0; i < GetDim(); i++){
        loc[i] = val[i];
    }
}

// Set element for a single point
void MappedData::SetElement( int vox_id, double * val ){
    float * loc = GetDataLocation(vox_id,true);
    for(int i = 0; i < GetDim(); i++){
        loc[i] = (float)val[i];
    }
}

// Wrappers for real GetElement functions
float              MappedData::GetElement( int x, int y, int z, int dim ) const {        return GetElement( GetVoxelID(x,y,z), dim );   }
std::vector<float> MappedData::GetElement( int x, int y, int z ) const {                 return GetElement( GetVoxelID(x,y,z) );        }
void               MappedData::GetElement( int x, int y, int z, float  * space ) const { return GetElement( GetVoxelID(x,y,z), space ); }
void               MappedData::GetElement( int x, int y, int z, double * space ) const { return GetElement( GetVoxelID(x,y,z), space ); }

// Get element returning a vector of floats
std::vector<float> MappedData::GetElement( int vox_id ) const {
    float * loc = GetDataLocation( vox_id );
    std::vector<float> ret;
    if(loc != 0){
        for(int i = 0; i < GetDim(); i++){
            ret.push_back( loc[i] );
        }
    }
    return ret;
}

// Get a single dimension from an element
float MappedData::GetElement( int vox_id, int dim ) const {
    float * loc = GetDataLocation( vox_id );
    if(loc != 0){
        return loc[dim];
    }
    return FLT_MAX;
}

// Get an element into a float array
void MappedData::GetElement( int vox_id, float  * space ) const {
    float * loc = GetDataLocation( vox_id );
    if(loc != 0){
        for(int i = 0; i < GetDim(); i++){
            space[i] = loc[i];
        }
    }
    else{
        for(int i = 0; i < GetDim(); i++){
            space[i] = FLT_MAX;
        }
    }
}

// Get an element into a doube array
void MappedData::GetElement( int vox_id, double * space ) const {
    float * loc = GetDataLocation( vox_id );
    if(loc != 0){
        for(int i = 0; i < GetDim(); i++){
            space[i] = loc[i];
        }
    }
    else{
        for(int i = 0; i < GetDim(); i++){
            space[i] = DBL_MAX;
        }
    }
}
*/
