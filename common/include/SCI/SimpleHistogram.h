#ifndef SIMPLEHISTOGRAM_H
#define SIMPLEHISTOGRAM_H

#include <vector>

namespace SCI {

    class SimpleHistogram {
    public:
        SimpleHistogram( int bins = 1, float minv = 0, float maxv = 1 );

        virtual void Reinitialize( int bins, float minv, float maxv );
        virtual void ClearBins();
        virtual void AddValue( float v );
        virtual void AddValues( std::vector<float> v );
        virtual int  GetBinCount();
        virtual int  GetBin(int j );

    protected:
        std::vector< int > bins;
        float minv, maxv;

    };

}

#endif // SIMPLEHISTOGRAM_H
