#include <RemoteDimensionalityReduction.h>

#include <string.h>

#include <iostream>
#include <limits>
#include <float.h>

using namespace std;


RemoteDimensionalityReduction::RemoteDimensionalityReduction(){
    connect( &tcpSocket, SIGNAL(connected()), this, SLOT(start()) );
}

void RemoteDimensionalityReduction::Start( int _method, Data::ProxyData & _data_in, SCI::Subset & _features, Data::BasicData & _data_out, SCI::Subset & _elements ){

    method   = _method;
    features = _features;
    elements = _elements;

    data_out = &_data_out;
    data_in  = &_data_in;

    featN    = 0;
    elemN    = 0;

    min_elem = SCI::VEX2_MAX;
    max_elem = SCI::VEX2_MIN;

    tcpSocket.abort();
    tcpSocket.connectToHost( data_in->GetIP(), data_in->GetPort() );

}

void RemoteDimensionalityReduction::run( ){
    tcpSocket.Send( "dr" );
    tcpSocket.Send( method );

    tcpSocket.Send( (int)features.size() );
    for(int i = 0; i < (int)features.size(); i++){
        tcpSocket.Send( features[i] );
    }

    if( data_in->GetVoxelCount() == (int)elements.size() ){
        tcpSocket.Send( -1 );
    }
    else{
        tcpSocket.Send( (int)elements.size() );
        for(int i = 0; i < (int)elements.size(); i++){
            tcpSocket.Send( elements[i] );
        }
    }
    tcpSocket.Send( "disconnect" );



    tcpSocket.setReadBufferSize( 4*1024*1024 );
    while( elemN < (int)elements.size() && tcpSocket.isOpen() ){
        SCI::Vex2 tmp[256];
        int t_len = features.size() + elements.size();
        int c_len = featN + elemN;
        int u_len = min( 256, t_len-c_len );

        tcpSocket.RecvBytes( (char*)tmp, sizeof(SCI::Vex2)*u_len, true );

        for(int i = 0; i < u_len; i++){
            if( featN < (int)features.size() ){
                data_out->SetElement( features[featN++], tmp[i].data );
            }
            else if( elemN < (int)elements.size() ){
                data_out->SetElement( elements[elemN++], tmp[i].data );
            }
            else{
                std::cout << "RemoteDimensionalityReduction: Extra data, don't know what to do with it..." << std::endl; fflush(stdout);
            }
            min_elem = SCI::Min( min_elem, tmp[i] );
            max_elem = SCI::Max( max_elem, tmp[i] );
        }
    }

    std::cout << "got " << elemN << " elements" << std::endl; fflush(stdout);

    if( tcpSocket.isOpen() ){
        tcpSocket.disconnectFromHost();
    }

    emit computationComplete(true);
}

void RemoteDimensionalityReduction::Restart( SCI::Subset & _elements ){

}

void RemoteDimensionalityReduction::Stop( ){

}


float RemoteDimensionalityReduction::GetMinValue( int dim ) const {
    if( dim < 0 || dim >= 2 ) return FLT_MAX;
    return min_elem[dim];
}

float RemoteDimensionalityReduction::GetMaxValue( int dim ) const {
    if( dim < 0 || dim >= 2 ) return FLT_MAX;
    return max_elem[dim];
}

int RemoteDimensionalityReduction::GetCurrentMethod() const {
    return method;
}


int RemoteDimensionalityReduction::GetFeatureCount() const {
    return featN;
}

int RemoteDimensionalityReduction::GetTotalFeatureCount() const {
    return features.size();
}

void RemoteDimensionalityReduction::GetFeaturePoint( int elem, float  * pout ) const {
    if(elem < featN) data_out->GetElement( features[elem], pout );
}

void RemoteDimensionalityReduction::GetFeaturePoint( int elem, double * pout ) const {
    if(elem < featN) data_out->GetElement( features[elem], pout );
}

int  RemoteDimensionalityReduction::GetFeatureVoxID( int elem ) const {
    return features[elem];
}

int RemoteDimensionalityReduction::GetTotalElementCount() const {
    return elements.size();
}

int RemoteDimensionalityReduction::GetElementCount() const {
    return elemN;
}

void RemoteDimensionalityReduction::GetElementPoint( int elem, float  * pout ) const {
    if(elem < elemN) data_out->GetElement( elements[elem], pout );
}

void RemoteDimensionalityReduction::GetElementPoint( int elem, double * pout ) const {
    if(elem < elemN) data_out->GetElement( elements[elem], pout );
}

int  RemoteDimensionalityReduction::GetElementVoxID( int elem ) const {
    return elements[elem];
}


