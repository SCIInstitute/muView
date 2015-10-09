#include <Data/Mesh/TetMesh.h>

#include <algorithm>
#include <QString>
#include <iostream>

using namespace Data;
using namespace Data::Mesh;

#include <Data/MultiDimensionalDataVector.h>


FILE * __open_write__( const char * fname, bool compress );
FILE * __open_read__( const char * fname, bool compressed );
void __close__( FILE * & file, bool compress );



Tetrahedron::Tetrahedron( int _v0, int _v1, int _v2, int _v3 ) : v0(_v0), v1(_v1), v2(_v2), v3(_v3) { }


Tetrahedron::Tetrahedron( FILE * file, bool binary ) : v0(-1), v1(-1), v2(-1), v3(-1) {
    if(binary){
        fread( data, sizeof(int), 4, file );
    }
    else{
        fscanf( file, " %i %i %i %i ",&v0,&v1,&v2,&v3 );
    }
}

Tetrahedron::Tetrahedron( const char * buf ) : v0(-1), v1(-1), v2(-1), v3(-1) {
    sscanf( buf, " %i %i %i %i ",&v0,&v1,&v2,&v3 );
}

bool Tetrahedron::isValid(){
    return ( v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0 );
}

float Tetrahedron::GetVolume( const PointMesh & pmesh ){
    SCI::Vex3 base = pmesh.GetVertex( v0 );
    SCI::Vex3 a = pmesh.GetVertex( v1 ) - base;
    SCI::Vex3 b = pmesh.GetVertex( v2 ) - base;
    SCI::Vex3 c = pmesh.GetVertex( v3 ) - base;
    return fabsf( SCI::dot(a, SCI::cross(b,c) ) ) / 6.0f;
}

void Tetrahedron::GetBoundingBox( SCI::Vex3 & bbmin, SCI::Vex3 & bbmax, const PointMesh & pmesh ){
    bbmin = bbmax = pmesh.GetVertex( data[0] );
    for(int i = 1; i < 4; i++){
        bbmin = SCI::Min( bbmin, pmesh.GetVertex( data[i] ) );
        bbmax = SCI::Max( bbmax, pmesh.GetVertex( data[i] ) );
    }
}

bool Tetrahedron::isPointContained( SCI::Vex3 p, const PointMesh & pmesh ){
    SCI::Vex3 v[4];
    for(int i = 0; i < 4; i++){
        v[i] = pmesh.GetVertex( data[i] );
    }

    for(int i = 0; i < 4; i++){
        SCI::Vex3 a = v[(i+0)%4];
        SCI::Vex3 b = v[(i+1)%4];
        SCI::Vex3 c = v[(i+2)%4];
        SCI::Vex3 d = v[(i+3)%4];
        SCI::Vex3 n = SCI::cross( b-a, c-a );
        if( SCI::dot(n,d-a) * SCI::dot(n,p-a) < 0 ) return false;
    }
    return true;
}

void Tetrahedron::ExtractSurface( TriMesh & tris, PointMesh & verts, const PointMesh & verts_in, float val0, float val1, float val2, float val3, float iso_val, const PointData * param_in, PointData * param_out ){

    //const std::vector<SCI::Vex3> * vert = &(pdata.vertices);
    SCI::Vex3 ppp[4][4];
    std::vector<float> prm[4][4];

    int type = 0;
    if( val0 >= iso_val ){ type += 1; }
    if( val1 >= iso_val ){ type += 2; }
    if( val2 >= iso_val ){ type += 4; }
    if( val3 >= iso_val ){ type += 8; }

    ppp[0][1] = ppp[1][0] = SCI::lerp( verts_in.points.at(v0), verts_in.points.at(v1), (iso_val-val0)/(val1-val0) );
    ppp[0][2] = ppp[2][0] = SCI::lerp( verts_in.points.at(v0), verts_in.points.at(v2), (iso_val-val0)/(val2-val0) );
    ppp[0][3] = ppp[3][0] = SCI::lerp( verts_in.points.at(v0), verts_in.points.at(v3), (iso_val-val0)/(val3-val0) );
    ppp[1][2] = ppp[2][1] = SCI::lerp( verts_in.points.at(v1), verts_in.points.at(v2), (iso_val-val1)/(val2-val1) );
    ppp[1][3] = ppp[3][1] = SCI::lerp( verts_in.points.at(v1), verts_in.points.at(v3), (iso_val-val1)/(val3-val1) );
    ppp[2][3] = ppp[3][2] = SCI::lerp( verts_in.points.at(v2), verts_in.points.at(v3), (iso_val-val2)/(val3-val2) );

    if( param_in ){
        prm[0][1] = prm[1][0] = SCI::lerp( param_in->GetElement(v0), param_in->GetElement(v1), (iso_val-val0)/(val1-val0) );
        prm[0][2] = prm[2][0] = SCI::lerp( param_in->GetElement(v0), param_in->GetElement(v2), (iso_val-val0)/(val2-val0) );
        prm[0][3] = prm[3][0] = SCI::lerp( param_in->GetElement(v0), param_in->GetElement(v3), (iso_val-val0)/(val3-val0) );
        prm[1][2] = prm[2][1] = SCI::lerp( param_in->GetElement(v1), param_in->GetElement(v2), (iso_val-val1)/(val2-val1) );
        prm[1][3] = prm[3][1] = SCI::lerp( param_in->GetElement(v1), param_in->GetElement(v3), (iso_val-val1)/(val3-val1) );
        prm[2][3] = prm[3][2] = SCI::lerp( param_in->GetElement(v2), param_in->GetElement(v3), (iso_val-val2)/(val3-val2) );
    }

    int vn = (int)verts.points.size();

    int me0 = -1, me1 = -1;
    if( type == 1 || (~type&0xf) == 1 ){ me0 = 0; }
    if( type == 2 || (~type&0xf) == 2 ){ me0 = 1; }
    if( type == 3 || (~type&0xf) == 3 ){ me0 = 0; me1 = 1; }
    if( type == 4 || (~type&0xf) == 4 ){ me0 = 2; }
    if( type == 5 || (~type&0xf) == 5 ){ me0 = 0; me1 = 2; }
    if( type == 6 || (~type&0xf) == 6 ){ me0 = 1; me1 = 2; }
    if( type == 8 || (~type&0xf) == 8 ){ me0 = 3; }


    if(me0 != -1 && me1 == -1){
        for(int i = 0; i < 4; i++){
            if(i!=me0){
                if( param_out ){
                    param_out->SetElement( (int)verts.points.size(), prm[me0][i] );
                }
                verts.points.push_back( ppp[me0][i] );
            }
        }
        Triangle tri( vn, vn+1, vn+2 );
        if( (~type&0xf) == 1 || type == 2 || (~type&0xf) == 4 || type == 8 ){ tri.Flip(); }
        tris.push_back( tri );
    }

    if(me0 != -1 && me1 != -1){
        for(int i = 0; i < 4; i++){
            if(i!=me0&&i!=me1){
                if( param_out ){
                    param_out->SetElement( (int)verts.points.size(), prm[me0][i] );
                }
                verts.points.push_back( ppp[me0][i] );
            }
        }
        for(int i = 0; i < 4; i++){
            if(i!=me0&&i!=me1){
                if( param_out ){
                    param_out->SetElement( (int)verts.points.size(), prm[me1][i] );
                }
                verts.points.push_back( ppp[me1][i] );
            }
        }
        Triangle tri0( vn,   vn+1, vn+2 );
        Triangle tri1( vn+1, vn+2, vn+3 );

        if( (~type&0xf) == 3 || type == 5 || (~type&0xf) == 6 ){ tri0.Flip(); }
        if( type == 3 || (~type&0xf) == 5 || type == 6 ){        tri1.Flip(); }

        tris.push_back( tri0 );
        tris.push_back( tri1 );
    }
}

#include <SCI/Geometry/CompGeom.h>

void  Tetrahedron::ExtractSurface( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out ){

    SCI::Vex3 p0 = verts_in.GetVertex( v0 );
    SCI::Vex3 p1 = verts_in.GetVertex( v1 );
    SCI::Vex3 p2 = verts_in.GetVertex( v2 );
    SCI::Vex3 p3 = verts_in.GetVertex( v3 );

    float d0 = RayPlaneIntersection( p0, p1-p0, plane );
    float d1 = RayPlaneIntersection( p0, p2-p0, plane );
    float d2 = RayPlaneIntersection( p0, p3-p0, plane );
    float d3 = RayPlaneIntersection( p1, p2-p1, plane );
    float d4 = RayPlaneIntersection( p1, p3-p1, plane );
    float d5 = RayPlaneIntersection( p2, p3-p2, plane );

    int cnt = 0;
    cnt += ( d0 >= 0 && d0 <= 1 ) ? 1 : 0;
    cnt += ( d1 >= 0 && d1 <= 1 ) ? 1 : 0;
    cnt += ( d2 >= 0 && d2 <= 1 ) ? 1 : 0;
    cnt += ( d3 >= 0 && d3 <= 1 ) ? 1 : 0;
    cnt += ( d4 >= 0 && d4 <= 1 ) ? 1 : 0;
    cnt += ( d5 >= 0 && d5 <= 1 ) ? 1 : 0;

    if( cnt == 0 ){ return; }

    if( cnt == 3 || cnt == 4 ){
        if( cnt == 3 ){
            tris_out.push_back( Triangle((int)verts_out.points.size(), (int)verts_out.points.size()+1, (int)verts_out.points.size()+2 ) );
        }
        if(cnt == 4){
            tris_out.push_back( Triangle((int)verts_out.points.size(), (int)verts_out.points.size()+1, (int)verts_out.points.size()+3 ) );
            tris_out.push_back( Triangle((int)verts_out.points.size(), (int)verts_out.points.size()+2, (int)verts_out.points.size()+3 ) );
        }

        if( d0 >= 0 && d0 <= 1 ){
            pdata_out.PushElement( SCI::lerp( pdata_in.GetElement(v0), pdata_in.GetElement(v1), d0) );
            verts_out.points.push_back( SCI::lerp(p0,p1,d0) );
        }
        if( d1 >= 0 && d1 <= 1 ){
            pdata_out.PushElement( SCI::lerp( pdata_in.GetElement(v0), pdata_in.GetElement(v2), d1) );
            verts_out.points.push_back( SCI::lerp(p0,p2,d1) );
        }
        if( d2 >= 0 && d2 <= 1 ){
            pdata_out.PushElement( SCI::lerp( pdata_in.GetElement(v0), pdata_in.GetElement(v3), d2) );
            verts_out.points.push_back( SCI::lerp(p0,p3,d2) );
        }
        if( d3 >= 0 && d3 <= 1 ){
            pdata_out.PushElement( SCI::lerp( pdata_in.GetElement(v1), pdata_in.GetElement(v2), d3) );
            verts_out.points.push_back( SCI::lerp(p1,p2,d3) );
        }
        if( d4 >= 0 && d4 <= 1 ){
            pdata_out.PushElement( SCI::lerp( pdata_in.GetElement(v1), pdata_in.GetElement(v3), d4) );
            verts_out.points.push_back( SCI::lerp(p1,p3,d4) );
        }
        if( d5 >= 0 && d5 <= 1 ){
            pdata_out.PushElement( SCI::lerp( pdata_in.GetElement(v2), pdata_in.GetElement(v3), d5) );
            verts_out.points.push_back( SCI::lerp(p2,p3,d5) );
        }
        return;
    }
    std::cerr << "Tetrahedron: don't know what to do with " << cnt << " intersections" << std::endl << std::flush;
}








TetMesh::TetMesh( ){

}

TetMesh::TetMesh(const char * tet_file){

    bool compressed = QString(tet_file).endsWith(".gz");

    if( QString(tet_file).endsWith(".btet") || QString(tet_file).endsWith(".btet.gz") ){
        //FILE * infile = fopen( tet_file, "rb" );
        FILE * infile = __open_read__( tet_file, compressed );
        if(infile){
            int tetN, triN;

            fread( &tetN,               sizeof(int),         1,                   infile );
            for(int i = 0; i < tetN; i++){
                tet_mesh.push_back( Tetrahedron( infile, true ) );
            }
            fread( &triN,              sizeof(int),         1,                   infile );
            for(int i = 0; i < triN; i++){
                tri_mesh.push_back( Triangle( infile, true ) );
            }

            //fclose( infile );
            __close__( infile, compressed );
        }

    }
    else{
        char buf[1024];
        FILE * infile = fopen(tet_file,"r");
        if(!infile) return;

        // first line might contain the number of elements instead of hex data, if it does skip it, if it doesn't add it
        fgets( buf, 1024, infile );
        if( Tetrahedron( buf ).isValid() ){
            tet_mesh.push_back( Tetrahedron( buf ) );
        }

        // process until the end of file
        do {
            tet_mesh.push_back( Tetrahedron(infile,false) );
        } while( tet_mesh.back().isValid() );

        // remove last invalid tet
        tet_mesh.pop_back();

        fclose(infile);

        RecalculateTrimesh();
    }

    printf("TetMesh: %i tets, %i tris\n",(int)tet_mesh.size(),(int)tri_mesh.size());
    fflush(stdout);

}

void TetMesh::Clear( ){
    tet_mesh.clear();
    tri_mesh.clear();
}

void TetMesh::BuildSpacePartition( const PointMesh & verts ){
    if( kdtree.isBuilt() ) return;
    printf("TetMesh: building kdtree\n"); fflush(stdout);
    SCI::Vex3 bmin,bmax;
    for(int i = 0; i < (int)tet_mesh.size(); i++){
        tet_mesh[i].GetBoundingBox( bmin, bmax, verts );
        kdtree.InsertElement( i, bmin, bmax );
    }
}

void TetMesh::ElementIndices( int elem, std::vector<int> & ind ){
    for(int i = 0; i < 4; i++){
        ind.push_back( tet_mesh[elem].data[i] );
    }
}

void TetMesh::RecalculateTrimesh( ){
    printf("TetMesh: calculating mesh\n"); fflush(stdout);

    int i[4];
    for(int j = 0; j < (int)tet_mesh.size(); j++){
        i[0] = tet_mesh[j].v0;
        i[1] = tet_mesh[j].v1;
        i[2] = tet_mesh[j].v2;
        i[3] = tet_mesh[j].v3;

        tri_mesh.push_back( Triangle( i[0],i[1],i[2] ) );
        tri_mesh.push_back( Triangle( i[0],i[1],i[3] ) );
        tri_mesh.push_back( Triangle( i[0],i[2],i[3] ) );
        tri_mesh.push_back( Triangle( i[1],i[2],i[3] ) );
    }
    tri_mesh.RemoveSharedFaces();
    printf("TetMesh: mesh complete\n"); fflush(stdout);
    printf("TetMesh: %i tets, %i tris\n",(int)tet_mesh.size(),(int)tri_mesh.size());  fflush(stdout);
}

bool TetMesh::Save(const char * fname){

    bool compress = QString(fname).endsWith(".gz");

    //FILE * outfile = fopen(fname,"wb");
    FILE * outfile = __open_write__(fname, compress );

    int tetN = (int)tet_mesh.size();
    int triN = (int)tri_mesh.size();

    if(outfile){
        fwrite( &tetN,                sizeof(int),         1,                   outfile );
        for(int i = 0; i < tetN; i++){
            fwrite( &(tet_mesh[i].data),     sizeof(int)*4,   1,     outfile );
        }
        fwrite( &triN,               sizeof(int),         1,                   outfile );
        for(int i = 0; i < triN; i++){
            fwrite( &(tri_mesh.at(i).data),     sizeof(int)*3,   1,     outfile );
        }
        //fclose( outfile );
        __close__(outfile, compress);
        return true;
    }

    return false;
}

float TetMesh::Volume( const PointMesh & verts ){
    std::cout << "TetMesh: " << "Finding volume" << std::endl << std::flush;
    double vol = 0;
    for(int i = 0; i < (int)tet_mesh.size(); i++){
        vol += tet_mesh[i].GetVolume( verts );
    }
    return (float)vol;
}



void TetMesh::ExtractIsotetrahedron( TetMesh & out, const PointData & pdata, float iso_val ){
    printf("TetMesh: extracting iso tetrahedron\n"); fflush(stdout);

    int thrdCnt = 4;

    std::vector< TetMesh* > tmesh;
    std::vector< TetMeshThread* > thrd;

    for(int i = 0; i < thrdCnt; i++){
        tmesh.push_back( new TetMesh() );
        thrd.push_back( new TetMeshThread(this) );
    }

    for(int i = 0; i < thrdCnt; i++){
        int s0 = (int)tet_mesh.size()*(i+0)/thrdCnt;
        int s1 = (int)tet_mesh.size()*(i+1)/thrdCnt;
        thrd[i]->ExtractIsotetrahedron( *tmesh[i], pdata, iso_val, s0, s1 );
    }

    for(int i = 0; i < thrdCnt; i++){
        thrd[i]->wait();
        for(int t = 0; t < (int)tmesh[i]->tet_mesh.size(); t++){
            out.tet_mesh.push_back( tmesh[i]->tet_mesh[t] );
        }
        delete thrd[i];
        delete tmesh[i];
    }
    out.RecalculateTrimesh();

    /*
    TetMesh t0, t1, t2, t3;
    TetMeshThread thrd0(this);
    TetMeshThread thrd1(this);
    TetMeshThread thrd2(this);
    TetMeshThread thrd3(this);
    int s0 = 0;
    int s1 = (int)tet_mesh.size()*1/4;
    int s2 = (int)tet_mesh.size()*2/4;
    int s3 = (int)tet_mesh.size()*3/4;
    int s4 = (int)tet_mesh.size();

    thrd0.ExtractIsotetrahedron( t0, pdata, iso_val, s0, s1 );
    thrd1.ExtractIsotetrahedron( t1, pdata, iso_val, s1, s2 );
    thrd2.ExtractIsotetrahedron( t2, pdata, iso_val, s2, s3 );
    thrd3.ExtractIsotetrahedron( t3, pdata, iso_val, s3, s4 );

    thrd0.wait();
    for(int t = 0; t < t0.tet_mesh.size(); t++){
        out.tet_mesh.push_back( t0.tet_mesh[t] );
    }
    thrd1.wait();
    for(int t = 0; t < t1.tet_mesh.size(); t++){
        out.tet_mesh.push_back( t1.tet_mesh[t] );
    }
    thrd2.wait();
    for(int t = 0; t < t2.tet_mesh.size(); t++){
        out.tet_mesh.push_back( t2.tet_mesh[t] );
    }
    thrd3.wait();
    for(int t = 0; t < t3.tet_mesh.size(); t++){
        out.tet_mesh.push_back( t3.tet_mesh[t] );
    }
    out.RecalculateTrimesh();
    */

    /*
    int dim = pdata.GetDim();
    std::vector<float> data[4];
    for(int i = 0; i < 4; i++){
        data[i].resize( dim );
    }
    for(int t = 0; t < (int)tet_mesh.size(); t++){
        Tetrahedron tet = tet_mesh[t];
        for(int i = 0; i < 4; i++){
            pdata.GetElement( tet.data[i], &(data[i][0])  );
        }

        bool lt = false;
        bool gt = false;
        for(int d = 0; d < dim; d++){
            for(int i = 0; i < 4; i++){
                lt = lt || ( data[i][d] <  iso_val );
                gt = gt || ( data[i][d] >= iso_val );
            }
            if(lt&&gt) break;
        }

        if( lt&&gt ){
            out.tet_mesh.push_back( tet );
        }
    }

    out.RecalculateTrimesh();
    */
}


void TetMesh::ExtractPlane( SCI::Vex4 plane, const PointMesh & verts_in, const PointData & pdata_in, TriMesh & tris_out, PointMesh & verts_out, MultiDimensionalDataVector & pdata_out ) {
    std::vector<int> elems;
    kdtree.FindPlaneIntersectionElements(plane,elems);
    for(int i = 0; i < elems.size(); i++){
        int t = elems[i];
        Tetrahedron tet = tet_mesh[t];
        tet.ExtractSurface( plane, verts_in, pdata_in, tris_out, verts_out, pdata_out );
    }
    verts_out.RecalculateBoundingBox();
}


void TetMesh::ExtractIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, int dim, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    for(int t = 0; t < (int)tet_mesh.size(); t++){
        Tetrahedron tet = tet_mesh[t];
        float val0 = pdata_in.GetElement( tet.v0, dim );
        float val1 = pdata_in.GetElement( tet.v1, dim );
        float val2 = pdata_in.GetElement( tet.v2, dim );
        float val3 = pdata_in.GetElement( tet.v3, dim );
        if( fabsf(val0) == FLT_MAX || fabsf(val1) == FLT_MAX || fabsf(val2) == FLT_MAX || fabsf(val3) == FLT_MAX ) continue;
        tet.ExtractSurface( tris_out, verts_out, verts_in, val0, val1, val2, val3, iso_val, param_in, param_out );
    }
    printf("TetMesh: Isosurface %i tris, %i verts\n",(int)tris_out.size(),(int)verts_out.points.size()); fflush(stdout);
}


void TetMesh::ExtractMinIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    for(int t = 0; t < (int)tet_mesh.size(); t++){
        Tetrahedron tet = tet_mesh[t];
        float val0 = pdata_in.GetMinimumValue( tet.v0 );
        float val1 = pdata_in.GetMinimumValue( tet.v1 );
        float val2 = pdata_in.GetMinimumValue( tet.v2 );
        float val3 = pdata_in.GetMinimumValue( tet.v3 );
        if( fabsf(val0) == FLT_MAX || fabsf(val1) == FLT_MAX || fabsf(val2) == FLT_MAX || fabsf(val3) == FLT_MAX ) continue;
        tet.ExtractSurface( tris_out, verts_out, verts_in, val0, val1, val2, val3, iso_val, param_in, param_out );
    }
    printf("TetMesh: Isosurface %i tris, %i verts\n",(int)tris_out.size(),(int)verts_out.points.size()); fflush(stdout);
}

void TetMesh::ExtractMeanIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    for(int t = 0; t < (int)tet_mesh.size(); t++){
        Tetrahedron tet = tet_mesh[t];
        float val0 = pdata_in.GetMeanValue( tet.v0 );
        float val1 = pdata_in.GetMeanValue( tet.v1 );
        float val2 = pdata_in.GetMeanValue( tet.v2 );
        float val3 = pdata_in.GetMeanValue( tet.v3 );
        if( fabsf(val0) == FLT_MAX || fabsf(val1) == FLT_MAX || fabsf(val2) == FLT_MAX || fabsf(val3) == FLT_MAX ) continue;
        tet.ExtractSurface( tris_out, verts_out, verts_in, val0, val1, val2, val3, iso_val, param_in, param_out );
    }
    printf("TetMesh: Isosurface %i tris, %i verts\n",(int)tris_out.size(),(int)verts_out.points.size()); fflush(stdout);
}

void TetMesh::ExtractMaxIsosurface(const PointMesh &verts_in, const PointData &pdata_in, float iso_val, TriMesh &tris_out, PointMesh &verts_out, PointData *param_in, PointData *param_out) const{
    for(int t = 0; t < (int)tet_mesh.size(); t++){
        Tetrahedron tet = tet_mesh[t];
        float val0 = pdata_in.GetMaximumValue( tet.v0 );
        float val1 = pdata_in.GetMaximumValue( tet.v1 );
        float val2 = pdata_in.GetMaximumValue( tet.v2 );
        float val3 = pdata_in.GetMaximumValue( tet.v3 );
        if( fabsf(val0) == FLT_MAX || fabsf(val1) == FLT_MAX || fabsf(val2) == FLT_MAX || fabsf(val3) == FLT_MAX ) continue;
        tet.ExtractSurface( tris_out, verts_out, verts_in, val0, val1, val2, val3, iso_val, param_in, param_out );
    }
    printf("TetMesh: Isosurface %i tris, %i verts\n",(int)tris_out.size(),(int)verts_out.points.size()); fflush(stdout);
}

int  TetMesh::ContainingElement( SCI::Vex3 p, const PointMesh & verts ){
    std::vector<int> elems;
    kdtree.FindContainingElements( p, elems );
    for(int j = 0; j < (int)elems.size(); j++){
        int i = elems[j];
        if( tet_mesh[i].isPointContained(p,verts) ){ return i; }
    }
    return -1;
}

void TetMesh::ExtractEdges( EdgeMesh & edges ){
    for(int i = 0; i < (int)tet_mesh.size(); i++){
        edges.push_back( Edge(tet_mesh[i].v0,tet_mesh[i].v1) );
        edges.push_back( Edge(tet_mesh[i].v0,tet_mesh[i].v2) );
        edges.push_back( Edge(tet_mesh[i].v0,tet_mesh[i].v3) );
        edges.push_back( Edge(tet_mesh[i].v1,tet_mesh[i].v2) );
        edges.push_back( Edge(tet_mesh[i].v1,tet_mesh[i].v3) );
        edges.push_back( Edge(tet_mesh[i].v2,tet_mesh[i].v3) );
    }
    edges.RemoveDuplicates();
    std::cout << "TetMesh: Extracted " << edges.size() << " edges" << std::endl << std::flush;
}




TetMeshThread::TetMeshThread( TetMesh * _parent ) : parent(_parent) { }

void TetMeshThread::ExtractIsotetrahedron( TetMesh & _out, const PointData & _pdata, float _iso_val, int _start, int _end ){
    out     = &_out;
    pdata   = &_pdata;
    iso_val = _iso_val;
    isotet  = true;
    t_start = _start;
    t_end   = _end;
    start();
}

void TetMeshThread::run(){

    if( isotet ){
        int dim = pdata->GetDim();
        std::vector<float> data[4];
        for(int i = 0; i < 4; i++){
            data[i].resize( dim );
        }
        for(int t = t_start; t < t_end && t < (int)parent->tet_mesh.size(); t++){
            Tetrahedron tet = parent->tet_mesh[t];
            for(int i = 0; i < 4; i++){
                pdata->GetElement( tet.data[i], &(data[i][0])  );
            }

            bool lt = false;
            bool gt = false;
            for(int d = 0; d < dim; d++){
                for(int i = 0; i < 4; i++){
                    lt = lt || ( data[i][d] <  iso_val );
                    gt = gt || ( data[i][d] >= iso_val );
                }
                if(lt&&gt) break;
            }

            if( lt&&gt ){
                out->tet_mesh.push_back( tet );
            }
        }
        isotet = false;
    }
}
