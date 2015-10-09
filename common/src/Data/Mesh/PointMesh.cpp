#include <Data/Mesh/PointMesh.h>

#include <iostream>
#include <algorithm>

#include <GL/oglCommon.h>

using namespace Data;
using namespace Data::Mesh;



#ifdef WIN32
    #define POPEN  _popen
    #define PCLOSE _pclose
#else
    #define POPEN  popen
    #define PCLOSE pclose
#endif


FILE * __open_write__( const char * fname, bool compress ){
    if( compress ){
        char cmd[1024];
        sprintf(cmd,"gzip > %s",fname);
        return POPEN( cmd, "w" );
    }
    return fopen(fname,"wb");
}

FILE * __open_read__( const char * fname, bool compressed ){
    if(compressed){
        char cmd[1024];
        sprintf(cmd,"gzip -dc %s",fname);
        return POPEN(cmd,"r");
    }
    return fopen(fname,"rb");
}

void __close__( FILE * & file, bool compress ){
    if(file){
        if(compress){
            PCLOSE(file);
        }
        else{
            fclose(file);
        }
        file = 0;
    }
}





PointMesh::PointMesh(){ }

PointMesh::PointMesh(const char * fname, bool binary , bool compressed){
    SCI::Vex3 vert;
    if( binary ){
        FILE * infile = __open_read__( fname, compressed );
        //FILE * infile = fopen( fname, "rb" );
        if(infile){
            int vsize;
            fread( &vsize, sizeof(int), 1, infile );
            for(int i = 0; i < vsize; i++){
                fread( &vert.data, sizeof(float)*3, 1, infile );
                points.push_back( vert );
            }
            //fclose( infile );
        }
        __close__( infile, compressed );
    }
    else{
        char buf[1024];
        FILE * infile;
        infile = fopen(fname,"r");
        if(infile){

            // first line might contain the number of elements instead of hex data, if it does skip it, if it doesn't add it
            fgets( buf, 1024, infile );
            if( sscanf( buf, " %f %f %f ",&vert.x,&vert.y,&vert.z) == 3 ){
                points.push_back(vert);
            }

            // process until the end of file
            while( fscanf( infile, " %f %f %f ",&vert.x,&vert.y,&vert.z) == 3 ){
                points.push_back(vert);
            }
            fclose(infile);
        }
    }
    std::cout << "PointMesh: " << points.size() << " points" << std::endl << std::flush;

    RecalculateBoundingBox();
}

void PointMesh::Clear(){
    points.clear();
    //colors.clear();
    //bbmin = SCI::VEX3_MAX;
    //bbmax = SCI::VEX3_MIN;
    bb.Reset();
}

void PointMesh::RecalculateBoundingBox(){

    //bbmin = SCI::VEX3_MAX;
    //bbmax = SCI::VEX3_MIN;
    bb.Reset();
    for(int i = 0; i < (int)points.size(); i++){
        //bbmin = SCI::Min( bbmin, points.at(i) );
        //bbmax = SCI::Max( bbmax, points.at(i) );
        bb.Expand( points.at(i) );
    }

    //std::cout << "PointMesh: Bounding Box ";
    //std::cout << "(" << bbmin.x << "," << bbmin.y << "," << bbmin.z << ") ";
    //std::cout << bb.bbmin << " " << bb.bbmax;
    //std::cout << "(" << bbmax.x << "," << bbmax.y << "," << bbmax.z << ") ";
    //std::cout << std::endl << std::flush;

}

void PointMesh::Save(const char * fname , bool compress){
    //FILE * outfile = fopen( fname, "wb" );
    FILE * outfile = __open_write__( fname, compress );
    if(outfile){
        int vsize = (int)points.size();
        fwrite( &vsize, sizeof(int), 1, outfile );
        for(int i = 0; i < vsize; i++){
            fwrite( &points[i].data, sizeof(float)*3, 1, outfile );
        }
        //fclose( outfile );
    }
    __close__( outfile, compress );
}

/*
void PointMesh::FillColors( unsigned int col ){
    while( colors.size() < points.size() ){
        colors.push_back(col);
    }
}
*/

SCI::Vex3 PointMesh::GetVertex( int i ) const {
    if( i < 0 || i >= (int)points.size() ){
        std::cout << "PointMesh: Invalid Index (" << i << ")" << std::endl << std::flush;
        return SCI::Vex3();
    }
    return points[i];
}

void PointMesh::Draw( ColorMesh & colors ){
    glBegin(GL_POINTS);
    for(int i = 0; i < (int)points.size(); i++){
        glColor4ubv( (GLubyte*)&colors.GetColor( i ) );
        glVertex3fv( points[i].data );
    }
    glEnd();
}

void PointMesh::Draw(SCI::Vex4 override_color){
    glColor4fv(override_color.data);
    glBegin(GL_POINTS);
    for(int i = 0; i < (int)points.size(); i++){
        glVertex3fv( points[i].data );
    }
    glEnd();
}


void PointMesh::FuseCoindidentVertices( std::map<int,int> & idx_map, int range_min, int range_max, float eps ){

    std::cout << "PointMesh: Fusing nearby vertices" << std::endl << std::flush;
    std::vector< SCI::Vex3 > new_verts;
    std::vector< unsigned int > new_cols;

    std::cout << "PointMesh: Sorting vertices" << std::endl << std::flush;
    std::vector< std::pair< float, int > > sorted_verts;
    for(int i = 0; i < (int)points.size(); i++){
        if( i < range_min || i > range_max ){
            idx_map[ i ] = (int)new_verts.size();
            new_verts.push_back( points[i] );
            //new_cols.push_back( colors[i] );
        }
        else{
            sorted_verts.push_back( std::make_pair( points[i].x, i ) );
        }
    }
    std::sort( sorted_verts.begin(), sorted_verts.end() );

    // worst case n^2, expected n * w, where w is a maximum window size relative to eps
    for(int k = 0; k < (int)sorted_verts.size(); k++){
        if( (k%100000)==99999 ){
            std::cout << "PointMesh: Fusing nearby vertices " << (k+1) << " of " << (int)sorted_verts.size() << std::endl << std::flush;
        }
        if( sorted_verts[k].first != FLT_MAX ){
            SCI::Vex4 ctmp;

            int i = sorted_verts[k].second;

            //ctmp.SetFromUInt( colors[ i ] );
            SCI::Vex3 npoint = points[ i ];
            //SCI::Vex4 ncolor = ctmp;
            int p_cnt = 1;
            idx_map[ i ] = (int)new_verts.size();

            for(int q = k+1; q < (int)sorted_verts.size() && ( sorted_verts[q].first - sorted_verts[k].first ) <= eps; q++){

                int j = sorted_verts[q].second;

                if( (points[i]-points[j]).Length() < eps ){
                    //ctmp.SetFromUInt( colors[j] );
                    npoint += points[j];
                    //ncolor += ctmp;
                    p_cnt++;
                    idx_map[ j ] = (int)new_verts.size();
                    points[j] = SCI::VEX3_INVALID;
                    sorted_verts[q].first = FLT_MAX;
                }
            }

            sorted_verts[k].first = FLT_MAX;

            new_verts.push_back( npoint / p_cnt );
            //new_cols.push_back( (ncolor/p_cnt).UIntColor() );
        }
    }

    std::cout << "PointMesh: Fusing reduced " << points.size() << " to " << new_verts.size() << " vertices" << std::endl << std::flush;


    points = new_verts;
    //colors = new_cols;


}
