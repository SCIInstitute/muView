#include <SCI/SimpleHistogram.h>
#include <SCI/Utility.h>

using namespace SCI;

SimpleHistogram::SimpleHistogram( int _bins, float _minv, float _maxv ){
    bins.resize(_bins);
    minv = _minv;
    maxv = _maxv;
    ClearBins();
}

void SimpleHistogram::Reinitialize( int _bins, float _minv, float _maxv ){
    bins.resize(_bins);
    minv = _minv;
    maxv = _maxv;
    ClearBins();
}

void SimpleHistogram::ClearBins(){
    for(int i = 0; i < bins.size(); i++){
        bins[i] = 0;
    }
}

void SimpleHistogram::AddValue( float v ){
    bins[ Quantize( v, minv, maxv, bins.size() ) ]++;
}

void SimpleHistogram::AddValues( std::vector<float> v ){
    for(int i = 0; i < v.size(); i++){
        bins[ Quantize( v[i], minv, maxv, bins.size() ) ];
    }
}

int  SimpleHistogram::GetBinCount(){
    return bins.size();
}

int  SimpleHistogram::GetBin(int j ){
    return bins[j%bins.size()];
}
