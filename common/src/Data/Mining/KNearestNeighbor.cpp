#include <Data/Mining/KNearestNeighbor.h>
#include <algorithm>
#include <math.h>

KNearestNeighbor::KNearestNeighbor(){
    input = output = 0;
}

void KNearestNeighbor::SetData( Data::PhysicsData & _input, Data::PhysicsData & _output ){
    input = &_input;
    output = &_output;
}

float VECTOR_L2Norm( const std::vector<float> & f0, const std::vector<float> & f1 ){
    float ret = 0;
    for(int i = 0; i < (int)f0.size(); i++){
        ret += (f0[i]-f1[i])*(f0[i]-f1[i]);
    }
    return sqrtf(ret);
}

std::vector<float> VECTOR_MADD( const std::vector<float> & f0, const std::vector<float> & f1, float s ){
    std::vector<float> ret;
    for(int i = 0; i < (int)f0.size(); i++){
        ret.push_back( f0[i] + f1[i] * s );
    }
    return ret;
}

std::vector<float> VECTOR_MUL( const std::vector<float> & f, float s ){
    std::vector<float> ret;
    for(int i = 0; i < (int)f.size(); i++){
        ret.push_back( f[i] * s );
    }
    return ret;
}

std::vector<float> KNearestNeighbor::Find( std::vector<float> point, int k ){
    std::vector<float> ret;
    if( input == 0 || output == 0 || (int)point.size() != input->GetDim() ){
        return ret;
    }

    std::vector< std::pair<float,int> > dist;
    for( int i = 0; i < input->GetElementCount(); i++ ){
        std::vector<float> tmp = input->GetElement( i );
        dist.push_back( std::make_pair( VECTOR_L2Norm( point, tmp ), i ) );
    }
    std::sort(dist.begin(),dist.end());

    ret.resize( output->GetDim(), 0 );
    float scl = 0.0f;
    for(int i = 0; i < k; i++){
        //ret = VECTOR_MADD(ret, output->GetElement(dist[i].second), 1.0f/(float)k );
        ret = VECTOR_MADD(ret, output->GetElement(dist[i].second), 1.0f/(0.0001f+dist[i].first) );
        scl += 1.0f/(0.0001f+dist[i].first);
    }
    return VECTOR_MUL(ret,1.0f/scl);
}

