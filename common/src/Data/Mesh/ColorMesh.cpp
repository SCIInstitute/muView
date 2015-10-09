#include <Data/Mesh/ColorMesh.h>
#include <Data/Mesh/OrientationMesh.h>
#include <Data/Mesh/SolidMesh.h>
#include <Data/MultiDimensionalData.h>

#include <Data/KMeansClustering.h>

#include <math.h>

#include <iostream>

using namespace Data::Mesh;


ColorMesh::ColorMesh(){
    def_color = 0xffffffff;
    cpv = cpe = false;
}

void ColorMesh::clear(){
    std::vector<unsigned int>::clear();
    def_color = 0xffffffff;
    cpv = cpe = false;
}

unsigned int & ColorMesh::GetColor( int i ){
    if( i < 0 || i >= (int)size() ) return def_color;
    return at(i);
}

const unsigned int & ColorMesh::GetColor( int i ) const {
    if( i < 0 || i >= (int)size() ) return def_color;
    return at(i);
}

      unsigned int & ColorMesh::DefaultColor( )       { return def_color; }
const unsigned int & ColorMesh::DefaultColor( ) const { return def_color; }

bool ColorMesh::isColorPerVertex() const { return cpv; }
bool ColorMesh::isColorPerElement() const { return cpe; }

void ColorMesh::SetColorPerVertex(){  cpv = true; cpe = false; }
void ColorMesh::SetColorPerElement(){ cpe = true; cpv = false; }

void ColorMesh::Fill( const PointMesh & pmesh, SCI::Vex4 color ){
    SetColorPerVertex();
    while( size() < pmesh.points.size() ){ push_back( color.UIntColor() ); }
}

void ColorMesh::Fill(TriMesh & tmesh, SCI::Vex4 color){
    SetColorPerElement();
    while( size() < tmesh.size() ){ push_back( color.UIntColor() ); }
}

void ColorMesh::SetFromOrientation( OrientationMesh & ort_mesh, SCI::Vex3 vd, float power ){
    SetColorPerVertex();

    resize( ort_mesh.size() );
    #pragma omp parallel shared(ort_mesh)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(int i = 0; i < (int)ort_mesh.size(); i++){
            float d = 0.0f;
            if( ort_mesh[i].Length() != 0 ){
                d = powf( 1.0f - fabsf( SCI::dot( vd, ort_mesh[i] ) ), power );
            }
            at(i) = SCI::Vex4( 0, 0, 0, d ).UIntColor();
        }
    }
}

void ColorMesh::SetByDataDimension( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap, int dim ){
    SetColorPerVertex();

    std::vector<float> vals( pdata.GetElementCount() );
    #pragma omp parallel shared(vals,pdata)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(int i = 0; i < pdata.GetElementCount(); i++){
            vals[i] = pdata.GetElement( i, dim );
        }
    }

    // Determine the colormap range
    SCI::BoundingRange bbox;
    for(int i = 0; i < pdata.GetElementCount(); i++){
        for(int d = 0; d < pdata.GetDimension(); d++){
            bbox.Expand( pdata.GetElement( i, d ) );
        }
    }
    cmap.SetRange( bbox.bbmin, bbox.bbmax );

    vals.push_back( pdata.GetMinimumValue() );
    vals.push_back( pdata.GetMaximumValue() );

    __SetByValueAndColormap__(vals, cmap);
}

void ColorMesh::SetByDataMean( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap ){
    SetColorPerVertex();

    std::vector<float> vals( pdata.GetElementCount() );
    #pragma omp parallel shared(vals,pdata)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(int i = 0; i < pdata.GetElementCount(); i++){
            vals[i] = pdata.GetElement( i ).Mean();
        }
    }

    // Determine the colormap range
    SCI::BoundingRange bbox;
    for( int i = 0; i < (int)vals.size(); i++){
        bbox.Expand( vals[i] );
    }
    cmap.SetRange( bbox.bbmin, bbox.bbmax );


    __SetByValueAndColormap__(vals, cmap);
}

void ColorMesh::SetByDataMin( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap ){
    SetColorPerVertex();

    std::vector<float> vals( pdata.GetElementCount() );
    #pragma omp parallel shared(vals,pdata)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(int i = 0; i < pdata.GetElementCount(); i++){
            vals[i] = pdata.GetElement( i ).Min();
        }
    }

    // Determine the colormap range
    SCI::BoundingRange bbox;
    for( int i = 0; i < (int)vals.size(); i++){
        bbox.Expand( vals[i] );
    }
    cmap.SetRange( bbox.bbmin, bbox.bbmax );


    __SetByValueAndColormap__(vals, cmap);
}

void ColorMesh::SetByDataMax( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap ){
    SetColorPerVertex();

    std::vector<float> vals( pdata.GetElementCount() );
    #pragma omp parallel shared(vals,pdata)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(int i = 0; i < pdata.GetElementCount(); i++){
            vals[i] = pdata.GetElement( i ).Max();
        }
    }

    // Determine the colormap range
    SCI::BoundingRange bbox;
    for( int i = 0; i < (int)vals.size(); i++){
        bbox.Expand( vals[i] );
    }
    cmap.SetRange( bbox.bbmin, bbox.bbmax );


    __SetByValueAndColormap__(vals, cmap);
}

void ColorMesh::SetByDataStdev( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap ){
    SetColorPerVertex();

    std::vector<float> vals( pdata.GetElementCount() );
    #pragma omp parallel shared(vals,pdata)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(int i = 0; i < pdata.GetElementCount(); i++){
            vals[i] = pdata.GetElement( i ).Stdev();
        }
    }

    // Determine the colormap range
    SCI::BoundingRange bbox;
    for( int i = 0; i < (int)vals.size(); i++){
        bbox.Expand( vals[i] );
    }
    cmap.SetRange( bbox.bbmin, bbox.bbmax );


    __SetByValueAndColormap__(vals, cmap);
}

void ColorMesh::SetByDataIsoRange( const Data::MultiDimensionalData & pdata, SCI::SequentialColormap & cmap, SCI::Vex4 mincol, SCI::Vex4 maxcol, float isoval ){
    SetColorPerVertex();

    resize( pdata.GetElementCount() );
    cmap.SetRange( 0, (float)pdata.GetDimension() );

    for(int i = 0; i < pdata.GetElementCount(); i++){
        float val = 0;
        for(int d = 0; d < pdata.GetDimension(); d++){
            val += (pdata.GetElement( i, d ) >= isoval) ? 1 : 0;
        }
        at(i) = cmap.GetColor( val ).UIntColor();
        if( val <= 0                    ){ at(i) = mincol.UIntColor(); }
        if( val >= pdata.GetDimension() ){ at(i) = maxcol.UIntColor(); }
    }

}

void ColorMesh::__SetByValueAndColormap__( std::vector< float > val, SCI::SequentialColormap & cmap ){

    // Calculate final colors
    resize( val.size() );
    #pragma omp parallel shared(val,cmap)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for( int i = 0; i < (int)val.size(); i++ ){
            at(i) = cmap.GetColor( val[i] ).UIntColor();
        }

    }
}

void ColorMesh::SetByDataClusterID(const KMeansClustering & km , SCI::CatagoricalColormap &cmap){

    //km.GetClusterID()

}
