#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

#include <SCI/Vex3.h>
#include <SCI/SimpleHistogram.h>

#include <GL/oglFont.h>

class Histogram : public SCI::SimpleHistogram
{
public:
    Histogram();

    //void Initialize( int bins, float minv, float maxv );
    //void Clear();
    virtual void Reinitialize( int bins, float minv, float maxv );
    virtual void ClearBins();

    virtual void AddValue( float v );
    virtual void AddValues( std::vector<float> v );
    //int  GetBinCount();
    //int  GetBin(int j );
    void SetColor( SCI::Vex3 col );

    void Draw(float u0, float v0, float u1, float v1, int dim , oglWidgets::oglFont *font, float apsect);
    void DrawValues( float u0, float v0, float u1, float v1 );

    bool operator < ( const Histogram right ) const ;

protected:
    std::vector< float > values;
    SCI::Vex3 color;
    //std::vector< int > bins;
    //float minv, maxv;

};

#endif // HISTOGRAM_H
