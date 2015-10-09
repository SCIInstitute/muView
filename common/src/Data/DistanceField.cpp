#include <Data/DistanceField.h>

#include <iostream>
#include <algorithm>

#include <Data/Mesh/TetMesh.h>
#include <Data/Mesh/HexMesh.h>

using namespace Data;

DistanceField::DistanceField( bool _min_iso, bool _max_iso, float _iso_val ) {

    min_iso = _min_iso;
    max_iso = _max_iso;
    iso_val = _iso_val;

}

DistanceField::DistanceField(FILE *infile){
    Load(infile);
}

DistanceField::~DistanceField(){ }

bool DistanceField::Load( FILE *  infile ){

    int siz;

    fread( &min_iso, sizeof(bool),  1, infile );
    fread( &max_iso, sizeof(bool),  1, infile );
    fread( &iso_val, sizeof(float), 1, infile );
    fread( &siz,     sizeof(int),   1, infile );

    vals.resize(siz);
    for(int i = 0; i < siz; i++){
        fread( &(vals[i]), sizeof(float), 1, infile );
    }

    return true;

}

void DistanceField::PrintInfo( ){
    if(min_iso) std::cout << "DistanceField: minimum isosurface" << std::endl;
    if(max_iso) std::cout << "DistanceField: maximum isosurface" << std::endl;
    std::cout << "DistanceField: isoval " << iso_val << std::endl;
}

bool DistanceField::Save( FILE * outfile ){

    int siz = (int)vals.size();

    fwrite( &min_iso, sizeof(bool),  1, outfile );
    fwrite( &max_iso, sizeof(bool),  1, outfile );
    fwrite( &iso_val, sizeof(float), 1, outfile );
    fwrite( &siz,     sizeof(int),   1, outfile );

    for(int i = 0; i < siz; i++){
        fwrite( &(vals[i]), sizeof(float), 1, outfile );
    }

    return true;
}


void DistanceField::Process(Data::Mesh::PointMesh *p_mesh, Data::Mesh::SolidMesh *s_mesh, Data::PointData *p_data){

    iso_tris.clear();
    iso_points.Clear();

    if( p_mesh == 0 && s_mesh == 0 && p_data == 0 ){ return; }

    if( min_iso ){
        std::cout << "DistanceField: Extracting Minimum Isosurface (val=" << iso_val << ")" << std::endl << std::flush;
        s_mesh->ExtractMinIsosurface( *p_mesh, *p_data, iso_val, iso_tris, iso_points );
    }
    if( max_iso ){
        std::cout << "DistanceField: Extracting Maximum Isosurface (val=" << iso_val << ")" << std::endl << std::flush;
        s_mesh->ExtractMaxIsosurface( *p_mesh, *p_data, iso_val, iso_tris, iso_points );
    }

    Data::Mesh::TetMesh iso_tet;
    Data::Mesh::HexMesh iso_hex;

    std::vector<int> elements;

    std::cout << "DistanceField: Extracting Isotetraherdon (val=" << iso_val << ")" << std::endl << std::flush;
    s_mesh->ExtractIsotetrahedron( iso_tet, *p_data, iso_val );
    for(int i = 0; i < iso_tet.GetElementCount(); i++){
        for(int j = 0; j < 4; j++){
            elements.push_back( iso_tet.tet_mesh[i].data[j] );
        }
    }

    std::cout << "DistanceField: Extracting Isohexahedron (val=" << iso_val << ")" << std::endl << std::flush;
    s_mesh->ExtractIsohexahedron( iso_hex, *p_data, iso_val );
    for(int i = 0; i < iso_hex.GetElementCount(); i++){
        for(int j = 0; j < 8; j++){
            elements.push_back( iso_hex.hex_mesh[i].data[j] );
        }
    }

    std::cout << "DistanceField: Sorting elements" << std::endl << std::flush;
    std::sort( elements.begin(), elements.end() );

    std::vector<int>::iterator it;
    it = std::unique( elements.begin(), elements.end() );
    elements.resize( it - elements.begin() );
    std::cout << "DistanceField: Calculating " << elements.size() << " elements" << std::endl << std::flush;

    //iso_points.FillColors( SCI::Vex4(1.0f,1.0f,0.0f,0.4f).UIntColor() );
    iso_tris.FuseCoindidentVertices( iso_points );
    iso_tris.BuildSpacePartition( iso_points );

    int s0 = 0;
    int s1 = (int)elements.size()/4;
    int s2 = (int)elements.size()*2/4;
    int s3 = (int)elements.size()*3/4;
    int s4 = (int)elements.size();

    vals.resize( p_data->GetVoxelCount(), FLT_MAX );
    DistanceFieldThread df0( p_mesh, iso_points, iso_tris, &vals[0], &elements[s0], s1-s0 );
    DistanceFieldThread df1( p_mesh, iso_points, iso_tris, &vals[0], &elements[s1], s2-s1 );
    DistanceFieldThread df2( p_mesh, iso_points, iso_tris, &vals[0], &elements[s2], s3-s2 );
    DistanceFieldThread df3( p_mesh, iso_points, iso_tris, &vals[0], &elements[s3], s4-s3 );

    /*
    int s0 = 0;
    int s1 = p_data->GetVoxelCount()/4;
    int s2 = p_data->GetVoxelCount()*2/4;
    int s3 = p_data->GetVoxelCount()*3/4;
    int s4 = p_data->GetVoxelCount();

    vals.resize( p_data->GetVoxelCount(), FLT_MAX );
    DistanceFieldThread df0( p_mesh, iso_points, iso_tris, &vals[0], s0, s1 );
    DistanceFieldThread df1( p_mesh, iso_points, iso_tris, &vals[0], s1, s2 );
    DistanceFieldThread df2( p_mesh, iso_points, iso_tris, &vals[0], s2, s3 );
    DistanceFieldThread df3( p_mesh, iso_points, iso_tris, &vals[0], s3, s4 );
    */

    df0.wait();
    df1.wait();
    df2.wait();
    df3.wait();
    /*
    for(int i = 0; i < p_data->GetVoxelCount(); i++){
        if( i==0 || ((i%25000)==24999) ) std::cout << "DistanceField: Processed " << i+1 << " of " << p_data->GetVoxelCount() << std::endl << std::flush;
        std::vector<int> elems;
        iso_tris.kdtree.FindClosestElements( p_mesh->points[i], elems );
        float val = FLT_MAX;
        for(int k = 0; k < elems.size(); k++){
            int j = elems[k];
            val = SCI::Min(val, iso_tris.tris[j].GetDistance( p_mesh->points[i], iso_points, true ) );
        }
        vals.push_back( val );
    }
    */

}



DistanceFieldSet::DistanceFieldSet( const char * fname ){
    Load( fname );
}

DistanceFieldSet::DistanceFieldSet(){ }

DistanceFieldSet::~DistanceFieldSet(){
    for(int i = 0; i < (int)df.size(); i++){
        delete df[i];
    }
}

DistanceField * DistanceFieldSet::Insert( FILE *  infile ){
    df.push_back( new DistanceField( infile ) );
    return df.back();
}

DistanceField * DistanceFieldSet::Insert( bool _min_iso, bool _max_iso, float _iso_val ){
    df.push_back( new DistanceField( _min_iso, _max_iso, _iso_val ) );
    return df.back();
}

bool DistanceFieldSet::Load( const char * fname ){
    FILE * ifile = fopen( fname, "rb" );
    if(ifile){
        int cnt;
        fread( &cnt, sizeof(int), 1, ifile );
        for(int i = 0; i < cnt; i++){
            df.push_back( new DistanceField( ifile ) );
            df.back()->PrintInfo();
        }
        fclose(ifile);
        std::cout << "DistanceFieldSet: Load success (" << fname << ")" << std::endl << std::flush;

        for(int i = 0; i < (int)df.size(); i++){
            for(int j = i+1; j < (int)df.size(); j++){
                if( df[i]->iso_val > df[j]->iso_val ){
                    std::swap( df[i], df[j] );
                }
                else if( df[i]->iso_val == df[j]->iso_val && df[i]->max_iso ){
                    std::swap( df[i], df[j] );
                }
            }
        }
        return true;
    }
    std::cout << "DistanceFieldSet: Load failed (" << fname << ")" << std::endl << std::flush;
    return false;
}

void DistanceFieldSet::ExtractMedianIsosurface( float isoval, const Data::Mesh::SolidMesh & mesh_in, const Data::Mesh::PointMesh & points_in, Data::Mesh::TriMesh & df_tris, Data::Mesh::PointMesh & df_points, Data::PointData & df_dist ){

    std::cout << "DistanceFieldSet: Extracting median isosurface for isoval " << isoval << std::endl << std::flush;

    Data::DistanceField * df0 = 0;
    Data::DistanceField * df1 = 0;

    for(int i = 0; i < (int)df.size(); i++){
        if( df[i]->iso_val == isoval && df[i]->min_iso ) df0 = df[i];
        if( df[i]->iso_val == isoval && df[i]->max_iso ) df1 = df[i];
    }

    if(df0==0||df1==0) return;

    df0->PrintInfo();
    df1->PrintInfo();

    Data::PointData diff( (int)points_in.points.size(), 1 );
    Data::PointData param_in((int) points_in.points.size(), 1 );

    for(int i = 0; i < (int)points_in.points.size(); i++){
        float val = df1->vals[i] - df0->vals[i];
        if( df0->vals[i] == FLT_MAX && df1->vals[i] == FLT_MAX ){
            val = FLT_MAX;
        }
        else if( df0->vals[i] != FLT_MAX && df1->vals[i] == FLT_MAX ){
            val = -FLT_MAX;
        }
        else if( df0->vals[i] == FLT_MAX && df1->vals[i] != FLT_MAX ){
            val = -FLT_MAX;
        }
        diff.SetElement( i, 0, val );
        param_in.SetElement( i, 0, df0->vals[i] );
    }
    mesh_in.ExtractIsosurface( points_in, diff, 0, 0, df_tris, df_points, &param_in, &df_dist );

}

bool DistanceFieldSet::Save( const char * fname ){
    FILE * ofile = fopen( fname, "wb" );
    if(ofile){
        int cnt = (int)df.size();
        fwrite( &cnt, sizeof(int), 1, ofile );
        for(int i = 0; i < cnt; i++){
            df[i]->Save( ofile );
        }
        fclose(ofile);
        std::cout << "DistanceFieldSet: Save success (" << fname << ")" << std::endl << std::flush;
        return true;
    }
    std::cout << "DistanceFieldSet: Save failed (" << fname << ")" << std::endl << std::flush;
    return false;
}




DistanceFieldThread::DistanceFieldThread( Data::Mesh::PointMesh *_p_mesh, Data::Mesh::PointMesh & _iso_points, Data::Mesh::TriMesh & _iso_tris, float * _vals, int * _elems, int _elemN ){

    iso_points = &_iso_points;
    iso_tris   = &_iso_tris;
    vals       = _vals;
    //v_start    = _start;
    //v_end      = _end;
    p_mesh     = _p_mesh;
    elems      = _elems;
    elemN      = _elemN;
    start();
}

DistanceFieldThread::~DistanceFieldThread(){
    terminate();
}

void DistanceFieldThread::run(){
    for(int itmp = 0; itmp < elemN; itmp++){
        if( itmp==0 || ((itmp%1000)==999) ) std::cout << "DistanceField: Processed " << itmp << " of " << elemN << std::endl << std::flush;
        int i = elems[itmp];
        std::vector<int> el;
        iso_tris->kdtree.FindClosestElements( p_mesh->points[i], el );
        float val = FLT_MAX;
        for(int k = 0; k < (int)el.size(); k++){
            int j = el[k];
            val = SCI::Min(val, iso_tris->at(j).GetDistance( p_mesh->points[i], *iso_points, true ) );
        }
        vals[i] = val;
    }
}

