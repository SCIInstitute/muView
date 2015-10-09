#include <Data/Mesh/HexMesh.h>

#include <algorithm>
#include <QString>
#include <iostream>

using namespace Data;
using namespace Data::Mesh;


Hexahedron::Hexahedron( int _v0, int _v1, int _v2, int _v3, int _v4, int _v5, int _v6, int _v7 ) : v0(_v0), v1(_v1), v2(_v2), v3(_v3), v4(_v4), v5(_v5), v6(_v6), v7(_v7) { }
Hexahedron::Hexahedron( ){
    for(int i = 0; i < 8; i++){
        data[i] = -1;
    }
}
Hexahedron::Hexahedron( FILE * infile, bool binary ){
    for(int i = 0; i < 8; i++){
        data[i] = -1;
    }
    if(binary){
        fread( data, sizeof(int), 8, infile );
    }
    else{
        fscanf( infile, " %i %i %i %i %i %i %i %i ",&v0,&v1,&v2,&v3,&v4,&v5,&v6,&v7 );
    }
}

Hexahedron::Hexahedron( const char * buf ){
    for(int i = 0; i < 8; i++){
        data[i] = -1;
    }
    sscanf( buf, " %i %i %i %i %i %i %i %i ",&v0,&v1,&v2,&v3,&v4,&v5,&v6,&v7 );
}

void Hexahedron::Write( FILE * outfile ){
    fwrite( data, sizeof(int), 8, outfile );
}

bool Hexahedron::isValid(){
    for(int i = 0; i < 8; i++){
        if( data[i] == -1 ) return false;
    }
    return true;
}

float Hexahedron::GetVolume( const PointMesh & pmesh ){
    float vol = 0;
    for(int i = 0; i < 5; i++){
        vol += GetTetrahedron( i ).GetVolume( pmesh );
    }
    return vol;
}

Tetrahedron Hexahedron::GetTetrahedron( int i ) const {
    if( i == 0 ) return Tetrahedron( data[0], data[1], data[2], data[5] );
    if( i == 1 ) return Tetrahedron( data[0], data[2], data[3], data[7] );
    if( i == 2 ) return Tetrahedron( data[4], data[5], data[7], data[0] );
    if( i == 3 ) return Tetrahedron( data[5], data[6], data[7], data[2] );
    if( i == 4 ) return Tetrahedron( data[0], data[2], data[5], data[7] );
    return Tetrahedron(-1,-1,-1,-1);
}

SCI::BoundingBox Hexahedron::GetBoundingBox( const PointMesh & pmesh ){
    SCI::BoundingBox bb;
    for(int i = 0; i < 8; i++){
        bb.Expand( pmesh.GetVertex( data[i] ) );
    }
    return bb;
}

Quadrilateral Hexahedron::GetQuadrilateral(int i){
    switch(i){
        case 0: return Quadrilateral( data[0], data[1], data[2], data[3] ) ;
        case 1: return Quadrilateral( data[4], data[5], data[6], data[7] ) ;
        case 2: return Quadrilateral( data[0], data[1], data[5], data[4] ) ;
        case 3: return Quadrilateral( data[1], data[2], data[6], data[5] ) ;
        case 4: return Quadrilateral( data[2], data[3], data[7], data[6] ) ;
        case 5: return Quadrilateral( data[3], data[0], data[4], data[7] ) ;
    }
    return Quadrilateral();
}

Edge Hexahedron::GetEdge( int i ){
    switch(i){
        case  0: return Edge( data[0], data[1] ) ;
        case  1: return Edge( data[1], data[2] ) ;
        case  2: return Edge( data[2], data[3] ) ;
        case  3: return Edge( data[3], data[0] ) ;
        case  4: return Edge( data[4], data[5] ) ;
        case  5: return Edge( data[5], data[6] ) ;
        case  6: return Edge( data[6], data[7] ) ;
        case  7: return Edge( data[7], data[4] ) ;
        case  8: return Edge( data[1], data[5] ) ;
        case  9: return Edge( data[2], data[6] ) ;
        case 10: return Edge( data[3], data[7] ) ;
        case 11: return Edge( data[0], data[4] ) ;
    }
    return Edge();
}





HexMesh::HexMesh( ){

}

HexMesh::HexMesh( const char * hex_file ){

    if( QString(hex_file).endsWith(".bhex") ){
        FILE * infile = fopen( hex_file, "rb" );
        if(infile){
            int tetN, triN;

            fread( &tetN,               sizeof(int),         1,                   infile );
            for(int i = 0; i < tetN; i++){
                hex_mesh.push_back( Hexahedron( infile, true ) );
            }

            fread( &triN,              sizeof(int),         1,                   infile );
            for(int i = 0; i < triN; i++){
                tri_mesh.push_back( Triangle( infile, true ) );
            }

            fclose( infile );
        }

    }
    else{

        char buf[1024];
        FILE * infile = fopen(hex_file,"r");
        if(!infile) return;

        // first line might contain the number of elements instead of hex data, if it does skip it, if it doesn't add it
        fgets( buf, 1024, infile );
        if( Hexahedron( buf ).isValid() ){
            hex_mesh.push_back( Hexahedron( buf ) );
        }
        // process until the end of file
        do {
            hex_mesh.push_back( Hexahedron( infile, false ) );
        } while( hex_mesh.back().isValid() );
        hex_mesh.pop_back();

        fclose(infile);

        RecalculateTrimesh();
    }

    printf("HexMesh: %i tets, %i tris\n",(int)hex_mesh.size(),(int)tri_mesh.size());
    fflush(stdout);

}

void HexMesh::ExtractPlane( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out ) {
    for(int i = 0; i < hex_mesh.size(); i++){
        for(int j = 0; j < 5; j++){
            hex_mesh[i].GetTetrahedron(j).ExtractSurface(plane,verts_in,pdata_in,tris_out,verts_out,pdata_out);
        }
    }
}

void HexMesh::BuildSpacePartition( const PointMesh & verts ){
    if( kdtree.isBuilt() ) return;
    printf("HexMesh: building kdtree\n"); fflush(stdout);
    //SCI::Vex3 bmin,bmax;
    for(int i = 0; i < (int)hex_mesh.size(); i++){
        //hex_mesh[i].GetBoundingBox( bmin, bmax, verts );
        //kdtree.InsertElement( i, bmin, bmax );
        SCI::BoundingBox bb = hex_mesh[i].GetBoundingBox( verts );
        kdtree.InsertElement( i, bb.bbmin, bb.bbmax );
    }
}


void HexMesh::RecalculateTrimesh( ){
    printf("HexMesh: calculating mesh\n"); fflush(stdout);

    for(int i = 0; i < (int)hex_mesh.size(); i++){
        for(int j = 0; j < 6; j++){
            quad_mesh.quads.push_back( hex_mesh[i].GetQuadrilateral(j) );
        }
    }

    quad_mesh.RemoveSharedFaces();
    quad_mesh.ConvertToTrimesh( tri_mesh );

}

int HexMesh::ContainingElement( SCI::Vex3 p, const PointMesh & verts ){
    std::vector<int> elems;
    kdtree.FindContainingElements( p, elems );
    for(int j = 0; j < (int)elems.size(); j++){
        int i = elems[j];
        for(int k = 0; k < 5; k++){
            if( hex_mesh.at(i).GetTetrahedron(k).isPointContained(p,verts) ){ return i; }
        }
    }
    return -1;
}


void HexMesh::ElementIndices( int elem, std::vector<int> & ind ){
    for(int i = 0; i < 8; i++){
        ind.push_back( hex_mesh[elem].data[i] );
    }
}

void HexMesh::Clear(){
    hex_mesh.clear();
    quad_mesh.quads.clear();
    tri_mesh.clear();
}

bool HexMesh::Save( const char * fname ){

    FILE * outfile = fopen(fname,"wb");

    int tetN = (int)hex_mesh.size();
    int triN = (int)tri_mesh.size();

    if(outfile==0) return false;

    fwrite( &tetN,                sizeof(int),         1,                   outfile );
    for(int i = 0; i < tetN; i++){
        hex_mesh[i].Write( outfile );
    }
    fwrite( &triN,               sizeof(int),         1,                   outfile );
    for(int i = 0; i < triN; i++){
        tri_mesh.at(i).Write( outfile );
    }
    fclose( outfile );
    return true;

}


float HexMesh::Volume( const PointMesh & verts ){
    double vol = 0;
    for(int i = 0; i < (int)hex_mesh.size(); i++){
        vol += hex_mesh[i].GetVolume( verts );
    }
    return (float)vol;
}

void HexMesh::ExtractIsohexahedron(HexMesh &out, const PointData &pdata, float iso_val){
    printf("HexMesh: extracting isohexahedron\n"); fflush(stdout);

    int dim = pdata.GetDim();
    std::vector<float> data[8];
    for(int i = 0; i < 8; i++){
        data[i].resize( dim );
    }
    for(int h = 0; h < (int)hex_mesh.size(); h++){
        Hexahedron hex = hex_mesh.at(h);

        for(int i = 0; i < 8; i++){
            pdata.GetElement( hex.data[i], &(data[i][0])  );
        }

        bool lt = false;
        bool gt = false;
        for(int d = 0; d < dim; d++){
            for(int i = 0; i < 8; i++){
                lt = lt || ( data[i][d] <  iso_val );
                gt = gt || ( data[i][d] >= iso_val );
            }
            if(lt&&gt) break;
        }

        if( lt&&gt ){
            out.hex_mesh.push_back( hex );
        }
    }
    out.RecalculateTrimesh();

}


#include <SCI/Graphics/MarchingCubes.h>


void HexMesh::isosurface_fill_values( int type, const Hexahedron & hex, const PointData & pdata, float vals_out[2][2][2], int dim ) const{
    if(type == 0){
        vals_out[0][0][0] = pdata.GetElement( hex.data[0], dim );
        vals_out[1][0][0] = pdata.GetElement( hex.data[1], dim );
        vals_out[1][1][0] = pdata.GetElement( hex.data[2], dim );
        vals_out[0][1][0] = pdata.GetElement( hex.data[3], dim );
        vals_out[0][0][1] = pdata.GetElement( hex.data[4], dim );
        vals_out[1][0][1] = pdata.GetElement( hex.data[5], dim );
        vals_out[1][1][1] = pdata.GetElement( hex.data[6], dim );
        vals_out[0][1][1] = pdata.GetElement( hex.data[7], dim );
        return;
    }
    if(type == 1){
        vals_out[0][0][0] = pdata.GetMinimumValue( hex.data[0] );
        vals_out[1][0][0] = pdata.GetMinimumValue( hex.data[1] );
        vals_out[1][1][0] = pdata.GetMinimumValue( hex.data[2] );
        vals_out[0][1][0] = pdata.GetMinimumValue( hex.data[3] );
        vals_out[0][0][1] = pdata.GetMinimumValue( hex.data[4] );
        vals_out[1][0][1] = pdata.GetMinimumValue( hex.data[5] );
        vals_out[1][1][1] = pdata.GetMinimumValue( hex.data[6] );
        vals_out[0][1][1] = pdata.GetMinimumValue( hex.data[7] );
        return;
    }
    if(type == 2){
        vals_out[0][0][0] = pdata.GetMeanValue( hex.data[0] );
        vals_out[1][0][0] = pdata.GetMeanValue( hex.data[1] );
        vals_out[1][1][0] = pdata.GetMeanValue( hex.data[2] );
        vals_out[0][1][0] = pdata.GetMeanValue( hex.data[3] );
        vals_out[0][0][1] = pdata.GetMeanValue( hex.data[4] );
        vals_out[1][0][1] = pdata.GetMeanValue( hex.data[5] );
        vals_out[1][1][1] = pdata.GetMeanValue( hex.data[6] );
        vals_out[0][1][1] = pdata.GetMeanValue( hex.data[7] );
        return;
    }
    if(type == 3){
        vals_out[0][0][0] = pdata.GetMaximumValue( hex.data[0] );
        vals_out[1][0][0] = pdata.GetMaximumValue( hex.data[1] );
        vals_out[1][1][0] = pdata.GetMaximumValue( hex.data[2] );
        vals_out[0][1][0] = pdata.GetMaximumValue( hex.data[3] );
        vals_out[0][0][1] = pdata.GetMaximumValue( hex.data[4] );
        vals_out[1][0][1] = pdata.GetMaximumValue( hex.data[5] );
        vals_out[1][1][1] = pdata.GetMaximumValue( hex.data[6] );
        vals_out[0][1][1] = pdata.GetMaximumValue( hex.data[7] );
        return;
    }
    std::cout << "HexMesh: " << "Unknown isosurface type" << std::endl << std::flush;
}

void HexMesh::isosurface_fill_verts( const Hexahedron & hex, const PointMesh & verts_in, SCI::Vex3 verts_out[2][2][2] ) const{
    verts_out[0][0][0] = verts_in.GetVertex( hex.data[0] );
    verts_out[1][0][0] = verts_in.GetVertex( hex.data[1] );
    verts_out[1][1][0] = verts_in.GetVertex( hex.data[2] );
    verts_out[0][1][0] = verts_in.GetVertex( hex.data[3] );
    verts_out[0][0][1] = verts_in.GetVertex( hex.data[4] );
    verts_out[1][0][1] = verts_in.GetVertex( hex.data[5] );
    verts_out[1][1][1] = verts_in.GetVertex( hex.data[6] );
    verts_out[0][1][1] = verts_in.GetVertex( hex.data[7] );
}

void HexMesh::isosurface_fill_params( const Hexahedron & hex, const PointData * verts_in, std::vector<float> verts_out[2][2][2] ) const{
    if( verts_in == 0 ) return ;
    verts_out[0][0][0] = verts_in->GetElement( hex.data[0] );
    verts_out[1][0][0] = verts_in->GetElement( hex.data[1] );
    verts_out[1][1][0] = verts_in->GetElement( hex.data[2] );
    verts_out[0][1][0] = verts_in->GetElement( hex.data[3] );
    verts_out[0][0][1] = verts_in->GetElement( hex.data[4] );
    verts_out[1][0][1] = verts_in->GetElement( hex.data[5] );
    verts_out[1][1][1] = verts_in->GetElement( hex.data[6] );
    verts_out[0][1][1] = verts_in->GetElement( hex.data[7] );
}

bool HexMesh::isosurface_need_compute( float vals[2][2][2], float iso_val ) const{
    // optimization -- determine if we can skip this element
    bool gt=false, lt=false;
    for(int x=0;x<2;x++){
        for(int y=0;y<2;y++){
            for(int z=0;z<2;z++){
                gt = gt || vals[x][y][z]>=iso_val;
                lt = lt || vals[x][y][z]<=iso_val;
            }
        }
    }
    return (lt&&gt);
}

void HexMesh::isosurface_process( int type, int dim, TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, const PointData & pdata, float iso_val, PointData *param_in, PointData *param_out ) const{
    SCI::Graphics::MarchingCubes mc;
    std::vector<int> indices;
    SCI::Vex3          verts_tmp[2][2][2];
    float              val_tmp[2][2][2];
    std::vector<float> param_tmp[2][2][2];
    int paramN = 0;

    for(int h = 0; h < (int)hex_mesh.size(); h++){
        Hexahedron hex = hex_mesh[h];
        isosurface_fill_values(type,hex,pdata,val_tmp,dim);
        if( isosurface_need_compute(val_tmp,iso_val) ){
            isosurface_fill_verts(  hex, verts_in, verts_tmp );
            mc.ProcessTriangles( (int)verts.points.size(), indices, val_tmp, iso_val );
            mc.ProcessVertices( verts.points, verts_tmp, val_tmp, iso_val );
            if( param_in ){
                std::vector< std::vector<float> > p_out;
                isosurface_fill_params( hex, param_in, param_tmp );
                mc.ProcessParameters( p_out, param_tmp, val_tmp, iso_val );
                for(int i = 0; i < (int)p_out.size(); i++){
                    param_out->SetElement( paramN++, p_out[i] );
                }
            }
        }
    }
    tris.Append( indices );
    printf("HexMesh: Isosurface %i tris, %i verts\n",(int)tris.size(),(int)verts.points.size()); fflush(stdout);
}





void HexMesh::ExtractIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, int dim, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    isosurface_process( 0, dim, tris_out, verts_out, verts_in, pdata_in, iso_val, param_in, param_out );
}


void HexMesh::ExtractMinIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    isosurface_process( 1, -1, tris_out, verts_out, verts_in, pdata_in, iso_val, param_in, param_out );
}

void HexMesh::ExtractMeanIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    isosurface_process( 2, -1, tris_out, verts_out, verts_in, pdata_in, iso_val, param_in, param_out );
}

void HexMesh::ExtractMaxIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    isosurface_process( 3, -1, tris_out, verts_out, verts_in, pdata_in, iso_val, param_in, param_out );
}


void HexMesh::ExtractEdges( EdgeMesh & edges ){
    for(int i = 0; i < (int)hex_mesh.size(); i++){
        for(int j = 0; j < 12; j++){
            edges.push_back( hex_mesh[i].GetEdge(j) );
        }
    }
    edges.RemoveDuplicates();
}




