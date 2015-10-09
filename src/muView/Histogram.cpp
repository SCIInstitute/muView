#include <muView/Histogram.h>

#include <GL/oglCommon.h>

Histogram::Histogram() : SCI::SimpleHistogram( 1, 0, 1 ) {

}

void Histogram::ClearBins(){
    SimpleHistogram::ClearBins();
    values.clear();
}

void Histogram::Reinitialize( int binN, float _minv, float _maxv ){
    values.clear();
    SimpleHistogram::Reinitialize(binN,_minv,_maxv);
}

/*
void Histogram::Initialize( int binN, float _minv, float _maxv ){
    bins.clear();
    bins.resize( binN, 0 );
    minv = _minv;
    maxv = _maxv;
}

void Histogram::Clear(){
    int binN = (int)bins.size();
    bins.clear();
    bins.resize( binN, 0 );
    values.clear();
}

void Histogram::AddValue( float v ){
    values.push_back( v );
    int bin = SCI::Quantize( v, minv, maxv, bins.size() );
    bins[bin]++;
}*/

void Histogram::AddValue( float v ){
    values.push_back( v );
    SimpleHistogram::AddValue(v);
}

void Histogram::AddValues( std::vector<float> v ){
    for(int i = 0; i < v.size(); i++){
        values.push_back(v[i]);
    }
    SimpleHistogram::AddValues(v);
}


//int  Histogram::GetBinCount(){ return (int)bins.size(); }
//int  Histogram::GetBin(int j ){ return bins[j%bins.size()]; }
void Histogram::SetColor( SCI::Vex3 col ){ color = col; }

bool Histogram::operator < ( const Histogram right ) const {
    for(int i = 0; i < (int)bins.size() && i < (int)right.bins.size(); i++){
        if( bins[i] < right.bins[i] ) return true;
        if( bins[i] > right.bins[i] ) return false;
    }
    return false;
}

void Histogram::Draw( float u0, float v0, float u1, float v1, int dim, oglWidgets::oglFont * font, float aspect ){
    glEnable(GL_DEPTH_TEST);

    glLineWidth(3.0f);
    glColor3f(0,0,0);
    glBegin( GL_LINE_LOOP );
        glVertex3f( u0, v0, 0 );
        glVertex3f( u1, v0, 0 );
        glVertex3f( u1, v1, 0 );
        glVertex3f( u0, v1, 0 );
    glEnd();

    glLineWidth(1.0f);
    glColor3f(0.4f,0.4f,0.4f);
    glBegin( GL_LINES );
    for(int j = 0; j < GetBinCount(); j++){
        float tu0 = SCI::lerp( u0, u1, (float)(j+0)/(float)( GetBinCount() ) );
        float tu1 = SCI::lerp( u0, u1, (float)(j+1)/(float)( GetBinCount() ) );
        float tv0 = v0;
        float tv1 = SCI::lerp( v0+0.01f, v1-0.01f, (float)GetBin(j)/(float)dim);

        glVertex3f( tu0, tv0, 0 ); glVertex3f( tu1, tv0, 0 );
        glVertex3f( tu1, tv0, 0 ); glVertex3f( tu1, tv1, 0 );
        glVertex3f( tu1, tv1, 0 ); glVertex3f( tu0, tv1, 0 );
    }
    glEnd();

    glColor3fv( color.data );
    glBegin( GL_QUADS );
    for(int j = 0; j < GetBinCount(); j++){
        float tu0 = SCI::lerp( u0, u1, (float)(j+0)/(float)(GetBinCount()) );
        float tu1 = SCI::lerp( u0, u1, (float)(j+1)/(float)(GetBinCount()) );
        float tv0 = v0;
        float tv1 = SCI::lerp( v0+0.01f, v1-0.01f, (float)GetBin(j)/(float)dim );

        glVertex3f( tu0, tv0, 0 );
        glVertex3f( tu1, tv0, 0 );
        glVertex3f( tu1, tv1, 0 );
        glVertex3f( tu0, tv1, 0 );
    }
    glEnd();

    if( font ){
        glPushMatrix();
        glColor3f(0,0,0);
        glTranslatef(u0-0.01,v0+0.01,0);
        glRotatef(90,0,0,1);
        glScalef(0.05*aspect,0.05,1);
        font->Print( "Cluster Histogram" );
        glPopMatrix();
    }
}

void Histogram::DrawValues( float u0, float v0, float u1, float v1 ){
    glEnable(GL_DEPTH_TEST);

    glLineWidth(3.0f);
    glColor3f(0,0,0);
    glBegin( GL_LINE_LOOP );
        glVertex3f( u0, v0, 0 );
        glVertex3f( u1, v0, 0 );
        glVertex3f( u1, v1, 0 );
        glVertex3f( u0, v1, 0 );
    glEnd();

    glLineWidth(1.0f);
    glColor3f(0.4f,0.4f,0.4f);
    glBegin( GL_LINE_STRIP );
    for(int j = 0; j < (int)values.size(); j++){
        float val = values[j];
        float tu = SCI::lerp( u0, u1, (float)j/(float)(values.size()-1) );
        float tv = SCI::lerp( v0, v1, (val-minv)/(maxv-minv)*0.9f + 0.05f );
        glVertex3f( tu, tv, 0 );
    }
    glEnd();

    glColor3fv( color.data );
    glBegin( GL_QUAD_STRIP );
    for(int j = 0; j < (int)values.size(); j++){
        float val = values[j];
        float tu = SCI::lerp( u0, u1, (float)j/(float)(values.size()-1) );
        float tv = SCI::lerp( v0, v1, (val-minv)/(maxv-minv)*0.9f + 0.05f );
        glVertex3f( tu, v0, 0 );
        glVertex3f( tu, tv, 0 );
    }
    glEnd();

}




