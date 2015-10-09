#include <Data/Mesh/QuadMesh.h>

#include <vector>
#include <algorithm>
#include <iostream>

using namespace Data::Mesh;

Quadrilateral::Quadrilateral( FILE * file, bool binary ){
    if(binary){
        fread( data, sizeof(int), 4, file );
    }
    else{
        std::cout << "Quadrilateral: Not yet implemented" << std::endl; fflush(stdout);
    }
}

Quadrilateral::Quadrilateral( int _v0, int _v1, int _v2, int _v3 ) : v0(_v0), v1(_v1), v2(_v2), v3(_v3) { }

bool Quadrilateral::operator < ( const Quadrilateral right ) const {

    int minv = -1;
    // take 4 passes, each time compare the next smallest value from each
    for(int pass = 0; pass < 4; pass++){
        int smallest_l = INT_MAX;
        int smallest_r = INT_MAX;
        for(int i = 0; i < 4; i++){
            if(       data[i] > minv ) smallest_l = SCI::Min( smallest_l,       data[i] );
            if( right.data[i] > minv ) smallest_r = SCI::Min( smallest_r, right.data[i] );
        }
        if( smallest_l < smallest_r ){ return true; }
        if( smallest_l > smallest_r ){ return false; }
        minv = smallest_l;
    }
    return false;
}

bool Quadrilateral::operator != ( const Quadrilateral right ) const {
    std::vector<int> vl;
    std::vector<int> vr;

    vl.push_back(v0);       vl.push_back(v1);       vl.push_back(v2);       vl.push_back(v3);
    vr.push_back(right.v0); vr.push_back(right.v1); vr.push_back(right.v2); vr.push_back(right.v3);

    std::sort(vl.begin(),vl.end());
    std::sort(vr.begin(),vr.end());

    for(int i = 0; i < 4; i++){
        if( vl[i] != vr[i] ){ return true; }
    }
    return false;
}

Triangle Quadrilateral::GetTriangle(int i){
    switch(i){
        case 0: return Triangle( v0, v1, v2 );
        case 1: return Triangle( v0, v2, v3 );
    }
    return Triangle();
}

Edge Quadrilateral::GetEdge(int i){
    if(i == 0){ return Edge(v0,v1); }
    if(i == 1){ return Edge(v1,v2); }
    if(i == 2){ return Edge(v2,v3); }
    if(i == 3){ return Edge(v3,v0); }
    return Edge();
}















QuadMesh::QuadMesh(){

}

void QuadMesh::RemoveSharedFaces( ){

    std::cout << "QuadMesh: removing internal quadrilateral faces" << std::endl << std::flush;

    if( quads.size() == 0 ) return;
    if( quads.size() == 1 ) return;

    std::vector< Quadrilateral > tmp_verts( quads.begin(), quads.end() );
    std::sort( tmp_verts.begin(), tmp_verts.end() );
    quads.clear();

    if( tmp_verts[0] != tmp_verts[1]){
        quads.push_back( tmp_verts[0] );
    }
    for(int i = 1; i < (int)tmp_verts.size()-1; i++){
        if( tmp_verts[i-1] != tmp_verts[i] && tmp_verts[i] != tmp_verts[i+1] ){
            quads.push_back( tmp_verts[i] );
        }
    }
    if( tmp_verts[tmp_verts.size()-2] != tmp_verts[tmp_verts.size()-1]){
        quads.push_back( tmp_verts[tmp_verts.size()-1] );
    }

    std::cout << "QuadMesh: reduced " << tmp_verts.size() << " quads to " << quads.size() << " quads" << std::endl << std::flush;

}

void QuadMesh::ConvertToTrimesh( TriMesh & tri_mesh ){
    for(int i = 0; i < (int)quads.size(); i++){
        tri_mesh.push_back( quads[i].GetTriangle(0) );
        tri_mesh.push_back( quads[i].GetTriangle(1) );
    }
}

void QuadMesh::ExtractEdges( EdgeMesh & edges ){
    for(int i = 0; i < (int)quads.size(); i++){
        for(int j = 0; j < 4; j++){
            edges.push_back( quads[i].GetEdge(j) );
        }
    }
    edges.RemoveDuplicates();
}
