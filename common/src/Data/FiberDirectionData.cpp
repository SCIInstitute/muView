#include <Data/FiberDirectionData.h>

#include <iostream>

using namespace Data;


FiberDirectionData::FiberDirectionData( const char * fname, bool binary ){
    if( binary ){
        int siz;
        SCI::Vex3 xyz;
        FILE * infile = fopen( fname, "rb" );
        fread( &siz, sizeof(int), 1, infile );
        for(int i = 0; i < siz; i++){
            fread( xyz.data, sizeof(float), 3, infile );
            fibs.push_back( xyz );
        }
        fclose(infile);
    }
    else{
        char buf[1024];
        float x,y,z;
        FILE * infile = fopen( fname, "r" );
        fgets( buf, 1024, infile);
        if( sscanf( buf, " %f %f %f ",&x,&y,&z) == 3){
            fibs.push_back( SCI::Vex3(x,y,z) );
        }
        while( fscanf(infile, " %f %f %f ",&x,&y,&z) == 3){
            fibs.push_back( SCI::Vex3(x,y,z) );
        }
        fclose(infile);
    }

    std::cout << "FiberDirectionData: Read " << fibs.size() << " directions" << std::endl << std::flush;
}

void FiberDirectionData::Save( const char * fname ){
    FILE * outfile = fopen(fname,"wb");
    int siz = (int)fibs.size();
    fwrite( &siz, sizeof(int), 1, outfile );
    for(int i = 0; i < siz; i++){
        fwrite(fibs[i].data, sizeof(float), 3, outfile );
    }
    fclose(outfile);

}
