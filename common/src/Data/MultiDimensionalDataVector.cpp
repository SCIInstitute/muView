/*
**  Common Data Library
**  Copyright (C) 2013  Paul Rosen
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <Data/MultiDimensionalDataVector.h>

#include <SCI/Utility.h>
#include <float.h>

using namespace Data;

MultiDimensionalDataVector::MultiDimensionalDataVector( int _elemN, int _dimN ) : MultiDimensionalData(_elemN,_dimN){
    data.resize( elemN * dimN, FLT_MAX );
}

void MultiDimensionalDataVector::Resize( int _elemN, int _dimN ){
    MultiDimensionalData::Resize(_elemN,_dimN);
    data.resize( elemN * dimN, FLT_MAX );
}

// Get a rough estimate of the size of the data contained in the class
int MultiDimensionalDataVector::GetDataSize() const {
    return (int)(data.size() * sizeof(float));
}

// Various functions for setting values
void MultiDimensionalDataVector::SetElement( int elem_id, const std::vector<float> & val ){
    if( elem_id < 0 || elem_id >= GetElementCount() ) return;
    for(int cur_dim = 0; cur_dim < GetDimension() && cur_dim < (int)val.size(); cur_dim++){
        data[ elem_id * GetDimension() + cur_dim ] = val[cur_dim];
    }
    min_val = max_val = FLT_MAX;
}

void MultiDimensionalDataVector::SetElement( int elem_id, int dim, float val ){
    if( elem_id < 0 || elem_id >= GetElementCount() ) return;
    data[ elem_id * GetDimension() + dim ] = val;
    min_val = max_val = FLT_MAX;
}

void MultiDimensionalDataVector::SetElement( int elem_id, const float  * val ){
    if( elem_id < 0 || elem_id >= GetElementCount() ) return;
    for(int cur_dim = 0; cur_dim < GetDimension(); cur_dim++){
        data[  elem_id * GetDimension() + cur_dim ] = val[cur_dim];
    }
    min_val = max_val = FLT_MAX;
}

void MultiDimensionalDataVector::SetElement( int elem_id, const double * val ){
    if( elem_id < 0 || elem_id >= GetElementCount() ) return;
    for(int cur_dim = 0; cur_dim < GetDimension(); cur_dim++){
        data[ elem_id * GetDimension() + cur_dim ] = (float)val[cur_dim];
    }
    min_val = max_val = FLT_MAX;
}

void MultiDimensionalDataVector::PushElement( const std::vector<float> & val ){
    int cur_dim = 0;
    while( cur_dim < GetDimension() && cur_dim < val.size() ){ data.push_back( val[cur_dim] ); cur_dim++; }
    while( cur_dim < GetDimension()                         ){ data.push_back( FLT_MAX );      cur_dim++; }
    min_val = max_val = FLT_MAX;
    elemN++;
}

void MultiDimensionalDataVector::PushElement( const float * val ){
    int cur_dim = 0;
    while( cur_dim < GetDimension() ){ data.push_back( val[cur_dim] ); cur_dim++; }
    min_val = max_val = FLT_MAX;
    elemN++;
}

void MultiDimensionalDataVector::PushElement( const double * val ){
    int cur_dim = 0;
    while( cur_dim < GetDimension() ){ data.push_back( (float)val[cur_dim] ); cur_dim++; }
    min_val = max_val = FLT_MAX;
    elemN++;
}


// Various functions for getting values
SCI::VexN MultiDimensionalDataVector::GetElement( int elem_id ) const {
    SCI::VexN ret( GetDimension() );
    if( elem_id < 0 || elem_id >= GetElementCount() ) return ret;
    for(int cur_dim = 0; cur_dim < GetDimension(); cur_dim++){
        ret[cur_dim] = data[ elem_id * GetDimension() + cur_dim ] ;
    }
    return ret;
}

float MultiDimensionalDataVector::GetElement( int elem_id, int dim ) const {
    if( elem_id < 0 || elem_id >= GetElementCount() ) return FLT_MAX;
    return data[  elem_id * GetDimension() + dim  ];
}

void MultiDimensionalDataVector::GetElement( int elem_id, float  * space ) const {
    if( elem_id < 0 || elem_id >= GetElementCount() ) return;
    for(int cur_dim = 0; cur_dim < GetDimension(); cur_dim++){
        space[cur_dim] = data[ elem_id * GetDimension() + cur_dim ];
    }
}

void MultiDimensionalDataVector::GetElement( int elem_id, double * space ) const {
    if( elem_id < 0 || elem_id >= GetElementCount() ) return;
    for(int cur_dim = 0; cur_dim < GetDimension(); cur_dim++){
        space[cur_dim] = data[ elem_id * GetDimension() + cur_dim ];
    }
}


void MultiDimensionalDataVector::RecalculateMinumumAndMaximum(){
    if( fabsf(min_val) == FLT_MAX || fabsf(max_val) == FLT_MAX ){
        min_val =  FLT_MAX;
        max_val = -FLT_MAX;
        for(int cur_dim = 0; cur_dim < GetDimension(); cur_dim++){
            min_dval[cur_dim] =  FLT_MAX;
            max_dval[cur_dim] = -FLT_MAX;
            for(int elem_id = 0; elem_id < GetElementCount(); elem_id++ ){
                min_dval[cur_dim] = SCI::Min( min_dval[cur_dim], data[ elem_id * GetDimension() + cur_dim ] );
                max_dval[cur_dim] = SCI::Max( max_dval[cur_dim], data[ elem_id * GetDimension() + cur_dim ] );
            }
            min_val = SCI::Min( min_val, min_dval[cur_dim] );
            max_val = SCI::Max( max_val, max_dval[cur_dim] );
        }
    }
}

float MultiDimensionalDataVector::GetMaximumValue( int dim ) {
    RecalculateMinumumAndMaximum();
    return MultiDimensionalData::GetMaximumValue(dim);
}

float MultiDimensionalDataVector::GetMinimumValue( int dim ) {
    RecalculateMinumumAndMaximum();
    return MultiDimensionalData::GetMinimumValue(dim);
}


