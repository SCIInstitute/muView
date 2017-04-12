#include <DimensionalityReduction.h>

#include <string.h>

#include <iostream>
#include <limits>
#include <float.h>

using namespace std;

int DimensionalityReduction::GetMethodCount(){
    return DR_WRAPPER::GetMethodCount();
}

const char * DimensionalityReduction::GetMethodName( int i ){
    return DR_WRAPPER::GetMethodName(i);
}


DimensionalityReduction::DimensionalityReduction(){
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

DimensionalityReduction::~DimensionalityReduction(){ }


double * DimensionalityReduction::GetPrincipalComponent(int component){

    return dr.GetPrincipalComponent(component);
}

void DimensionalityReduction::Stop( ){ }


int DimensionalityReduction::GetElementCount() {
    processNextElement( );
    return e_comp;
}

int DimensionalityReduction::GetTotalElementCount() const {
    return elements->size();
}

void DimensionalityReduction::GetElementPoint( int elem, float  * pout ) const {
    if(elem < e_comp) data_out->GetElement( elements->at(elem), pout );
}

void DimensionalityReduction::GetElementPoint( int elem, double * pout ) const {
    if(elem < e_comp) data_out->GetElement( elements->at(elem), pout );
}

int  DimensionalityReduction::GetFeatureVoxID( int elem ) const {
    return features->at(elem);
}

int  DimensionalityReduction::GetElementVoxID( int elem ) const {
    return elements->at(elem);
}

void DimensionalityReduction::processNextElement(){

    if( e_comp == (int)elements->size() ) return;

    double * space_in  = new double[high_dimN];
    double * space_out = new double[low_dimN];

    data_in->GetElement( elements->at(e_comp), space_in );
    GetMappedPoint( space_in, space_out );
    data_out->SetElement( elements->at(e_comp), space_out );
    for(int d = 0; d < low_dimN; d++){
        min_elem[d] = SCI::Min( min_elem[d], (float)space_out[d] );
        max_elem[d] = SCI::Max( max_elem[d], (float)space_out[d] );
    }
    e_comp++;

    delete space_in;
    delete space_out;

}



void DimensionalityReduction::Start( int method, Data::MultiDimensionalData & _data_in, SCI::Subset & _features, Data::MultiDimensionalData & _data_out, SCI::Subset & _elements ){
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
        data_in->GetElement( features->at(i), tmp ); // tmp size = dimension size = number of runs/simulations
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


    for(int i = 0; i < 10; i++){
        processNextElement();
    }

}


void DimensionalityReduction::StartAxis( int method, Data::MultiDimensionalData & _data_in, SCI::Subset & _features, Data::MultiDimensionalData & _data_out, SCI::Subset & _elements ){
    std::cout << "DimensionalityReduction::StartAxis TODO" << std::endl;

    Stop();

    features = &_features;
    elements = &_elements;
    data_in  = &_data_in;
    data_out = &_data_out;
    e_comp   = 0;


    // don't do it on all vertices
    high_dimN     = features->size();//data_in->GetElementCount();
    low_dimN      = data_out->GetDimension();

    // TODO update size here
    dr.SetSize( data_in->GetDimension(), low_dimN, high_dimN );

    double *tmp;
    tmp = new double[high_dimN];
    for(int i = 0; i < (int) data_in->GetDimension(); i++){//loop over all dimensions = runs, simulations
        data_in->GetDimensionData( i, _features, tmp ); // New function here GetDimensionValues instead of Element. Only get vertices in features subset.
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

    /*
    for(int i = 0; i < 10; i++){
        processNextElement();
    }
    */

}

void DimensionalityReduction::Restart( SCI::Subset & _elements ){
    Stop();

    elements = &_elements;
    e_comp = 0;

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

    for(int i = 0; i < 10; i++){
        processNextElement();
    }

    //start();

}

void DimensionalityReduction::GetFeaturePoint( int elem, double * pout ) const {
    dr.GetOutputData( elem, pout );
}

void DimensionalityReduction::GetFeaturePoint( int elem, float * pout ) const {
    dr.GetOutputData( elem, pout );
}

void DimensionalityReduction::GetMappedPoint( float  * pin, float  * pout ) const {
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

void DimensionalityReduction::GetMappedPoint( double * pin, double * pout ) const {
    dr.GetMappedPoint( pin, pout );
}

int DimensionalityReduction::GetFeatureCount() const {
    if(features) return features->size();
    return 0;
}

int DimensionalityReduction::GetHighDimension() const {
    return high_dimN;
}

int DimensionalityReduction::GetLowDimension() const {
    return low_dimN;
}

float DimensionalityReduction::GetMinValue( int dim ) const {
    if( dim < 0 || dim >= low_dimN ) return FLT_MAX;
    return min_elem[dim];
}

float DimensionalityReduction::GetMaxValue( int dim ) const {
    if( dim < 0 || dim >= low_dimN ) return FLT_MAX;
    return max_elem[dim];
}

