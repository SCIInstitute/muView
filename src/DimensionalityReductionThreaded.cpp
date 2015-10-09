#include <DimensionalityReductionThreaded.h>

#include <string.h>

#include <iostream>
#include <limits>
#include <float.h>

using namespace std;

int DimensionalityReductionThreaded::GetMethodCount(){
    return DR_WRAPPER::GetMethodCount();
}

const char * DimensionalityReductionThreaded::GetMethodName( int i ){
    return DR_WRAPPER::GetMethodName(i);
}


DimensionalityReductionThreaded::DimensionalityReductionThreaded(){
    high_dimN     = 0;
    low_dimN      = 0;
    cur_method    = 0;
    e_comp        = 0;
    features = 0;
    elements = 0;
    data_in  = 0;
    data_out = 0;
    min_elem = 0;
    max_elem = 0;
}

DimensionalityReductionThreaded::~DimensionalityReductionThreaded(){
    Stop();
}


void DimensionalityReductionThreaded::Stop( ){
    should_terminate = true;
    wait();
    should_terminate = false;
}


int DimensionalityReductionThreaded::GetElementCount() const {
    return e_comp;
}

int DimensionalityReductionThreaded::GetTotalElementCount() const {
    return elements->size();
}

void DimensionalityReductionThreaded::GetElementPoint( int elem, float  * pout ) const {
    if(elem < e_comp) data_out->GetElement( elements->at(elem), pout );
}

void DimensionalityReductionThreaded::GetElementPoint( int elem, double * pout ) const {
    if(elem < e_comp) data_out->GetElement( elements->at(elem), pout );
}

int  DimensionalityReductionThreaded::GetFeatureVoxID( int elem ) const {
    return features->at(elem);
}

int  DimensionalityReductionThreaded::GetElementVoxID( int elem ) const {
    return elements->at(elem);
}

void DimensionalityReductionThreaded::run(){

    double * space_in  = new double[high_dimN];
    double * space_out = new double[low_dimN];

    for(int i = 0; i < (int)elements->size() && !should_terminate; i++){
        data_in->GetElement( elements->at(i), space_in );
        GetMappedPoint( space_in, space_out );
        data_out->SetElement( elements->at(i), space_out );
        for(int d = 0; d < low_dimN; d++){
            min_elem[d] = SCI::Min( min_elem[d], (float)space_out[d] );
            max_elem[d] = SCI::Max( max_elem[d], (float)space_out[d] );
        }
        e_comp = i+1;
    }

    delete space_in;
    delete space_out;

    emit computationComplete( should_terminate );

}

void DimensionalityReductionThreaded::Start( int method, Data::MultiDimensionalData & _data_in, SCI::Subset & _features, Data::MultiDimensionalData & _data_out, SCI::Subset & _elements ){
    Stop();

    features = &_features;
    elements = &_elements;
    data_in  = &_data_in;
    data_out = &_data_out;
    e_comp   = 0;

    high_dimN     = data_in->GetDimension();
    low_dimN      = data_out->GetDimension();

    dr.SetSize( features->size(), low_dimN, high_dimN );

    double *tmp;

    tmp = new double[high_dimN];
    for(int i = 0; i < (int)features->size(); i++){
        data_in->GetElement( features->at(i), tmp );
        dr.SetInputData( i, tmp );
    }
    delete [] tmp;

    if(min_elem) delete [] min_elem;
    if(max_elem) delete [] max_elem;
    min_elem = new float[low_dimN];
    max_elem = new float[low_dimN];

    cur_method = method;
    dr.SetMethod( method );
    dr.Run( );

    for(int d = 0; d < low_dimN; d++){
        min_elem[d] =  FLT_MAX;
        max_elem[d] = -FLT_MAX;
    }

    tmp = new double[low_dimN];
    for(int i = 0; i < (int)features->size(); i++){
        dr.GetOutputData( i, tmp );
        for(int d = 0; d < low_dimN; d++){
            min_elem[d] = SCI::Min( min_elem[d], (float)tmp[d] );
            max_elem[d] = SCI::Max( max_elem[d], (float)tmp[d] );
        }
    }
    delete [] tmp;

    start();
}

void DimensionalityReductionThreaded::Restart( SCI::Subset & _elements ){
    Stop();

    elements = &_elements;

    for(int d = 0; d < low_dimN; d++){
        min_elem[d] =  FLT_MAX;
        max_elem[d] = -FLT_MAX;
    }

    double * tmp;
    tmp = new double[low_dimN];
    for(int i = 0; i < (int)features->size(); i++){
        dr.GetOutputData( i, tmp );
        for(int d = 0; d < low_dimN; d++){
            min_elem[d] = SCI::Min( min_elem[d], (float)tmp[d] );
            max_elem[d] = SCI::Max( max_elem[d], (float)tmp[d] );
        }
    }
    delete [] tmp;


    start();

}

void DimensionalityReductionThreaded::GetFeaturePoint( int elem, double * pout ) const {
    dr.GetOutputData( elem, pout );
}

void DimensionalityReductionThreaded::GetFeaturePoint( int elem, float * pout ) const {
    dr.GetOutputData( elem, pout );
}

void DimensionalityReductionThreaded::GetMappedPoint( float  * pin, float  * pout ) const {
    double * din  = new double[high_dimN];
    double * dout = new double[low_dimN];
    for(int i = 0; i < high_dimN; i++){
        din[i]=pin[i];
    }
    GetMappedPoint(din,dout);
    for(int i = 0; i < low_dimN; i++){
        pout[i]=dout[i];
    }
}

void DimensionalityReductionThreaded::GetMappedPoint( double * pin, double * pout ) const {
    dr.GetMappedPoint( pin, pout );
}

int DimensionalityReductionThreaded::GetFeatureCount() const {
    if(features) return features->size();
    return 0;
}

int DimensionalityReductionThreaded::GetHighDimension() const {
    return high_dimN;
}

int DimensionalityReductionThreaded::GetLowDimension() const {
    return low_dimN;
}

float DimensionalityReductionThreaded::GetMinValue( int dim ) const {
    if( dim < 0 || dim >= low_dimN ) return FLT_MAX;
    return min_elem[dim];
}

float DimensionalityReductionThreaded::GetMaxValue( int dim ) const {
    if( dim < 0 || dim >= low_dimN ) return FLT_MAX;
    return max_elem[dim];
}


