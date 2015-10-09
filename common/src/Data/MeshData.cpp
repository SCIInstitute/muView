#include <Data/MeshData.h>

using namespace Data;

MeshData::MeshData( const char *conn_file, const char *geom_file, const char *data_file) : VolData( 1,1,1,1 ) {
    FILE * infile;
    infile = fopen(conn_file,"r");
    if(infile){
        float f0,f1,f2;
        while( fscanf( infile, " %f %f %f ",&f0,&f1,&f2) == 3 ){
            connectivity.push_back((int)f0);
            connectivity.push_back((int)f1);
            connectivity.push_back((int)f2);
        }
        fclose(infile);
    }

    infile = fopen(geom_file,"r");
    if(infile){
        float f0,f1,f2;
        while( fscanf( infile, " %f %f %f ",&f0,&f1,&f2) == 3 ){
            vertices.push_back(SCI::Vex3(f0,f1,f2));
        }
        fclose(infile);
    }

    infile = fopen(data_file,"r");
    std::vector<float> flist;
    if(infile){
        float f;
        while( fscanf( infile, " %f ",&f) == 1 ){
            flist.push_back(f);
        }
        fclose(infile);
    }

    __resize( (int)vertices.size(), 1, 1, (int)flist.size()/(int)vertices.size() );

    for(int i = 0; i < (int)flist.size(); i++){
        SetElement( i/dim, i%dim, flist[i] );
    }

    printf("%i vertices, %i triangles, %i dim\n",(int)vertices.size(),(int)connectivity.size()/3,dim);
    fflush(stdout);
}

MeshData::MeshData( const char * mesh_file ) : VolData(1,1,1,1) {
    FILE * infile = fopen( mesh_file, "rb" );
    if(infile){
        int _sx,_dim,cs;
        fread( &_sx,               sizeof(int),         1,                   infile );
        fread( &_dim,              sizeof(int),         1,                   infile );
        fread( &cs,                sizeof(int),         1,                   infile );
        connectivity.resize(cs,0);
        vertices.resize(_sx);
        //__resize( _sx,1,1,_dim );
        __resize( _sx,1,1,2000 );
        std::vector<float> tmp(_dim);
        fread( &(connectivity[0]), sizeof(int),         connectivity.size(), infile );
        fread( &(vertices[0]),     sizeof(SCI::Vex3), vertices.size(),     infile );
        //fread( vol,                sizeof(float),       sx*dim,              infile );
        for(int x = 0; x < sx; x++){
            fread( &(tmp[0]), sizeof(float), _dim, infile);
            SetElement( x, &(tmp[0]) );
        }
        fclose( infile );
    }

    printf("%i vertices, %i triangles, %i dim\n",(int)vertices.size(),(int)connectivity.size()/3,dim);
    fflush(stdout);
}

void MeshData::Save( const char * fname ){

    FILE * outfile = fopen(fname,"wb");

    if(outfile){
        int cs = (int)connectivity.size();
        fwrite( &sx,                sizeof(int),         1,                   outfile );
        fwrite( &dim,               sizeof(int),         1,                   outfile );
        fwrite( &cs,                sizeof(int),         1,                   outfile );
        fwrite( &(connectivity[0]), sizeof(int),         connectivity.size(), outfile );
        fwrite( &(vertices[0]),     sizeof(SCI::Vex3), vertices.size(),     outfile );
        fwrite( vol,                sizeof(float),       sx*dim,              outfile );
        fclose( outfile );
    }
}

std::vector<int>         & MeshData::GetConnectivity(){ return connectivity; }
std::vector<SCI::Vex3> & MeshData::GetVertices(){ return vertices; }
