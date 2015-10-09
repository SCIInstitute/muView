#include <Data/PointData.h>
#include <iostream>
#include <SCI/Utility.h>
#include <float.h>

using namespace Data;



PointData::PointData( int elem, int dim ) : DenseMultiDimensionalData( elem, dim ) { }

PointData::PointData( const char * filename, bool binary ) : DenseMultiDimensionalData(1,1) {
    fname = filename;
    bool load_ok = true;

    if( binary ){
        FILE * infile = fopen( filename, "rb" );
        if(infile){
            int _sx,_dim;
            load_ok = load_ok && ( fread( &_sx,  sizeof(int), 1, infile ) == 1);
            load_ok = load_ok && ( fread( &_dim, sizeof(int), 1, infile ) == 1);
            Resize( _sx,_dim );
            load_ok = load_ok && ( fread( &(data[0]), sizeof(float), GetElementCount()*GetDimension(), infile ) == ( GetElementCount()*GetDimension() ) );
            if( !load_ok ){ std::cerr << "PointData: Load failed on data" << std::endl << std::flush; }
            fclose( infile );
        }

    }
    else{
        std::cout << "PointData: loading " << filename << std::endl << std::flush;
        FILE * infile = fopen(filename,"r");
        if(infile){
            std::vector<float> flist;
            float f;
            while( fscanf( infile, " %f ",&f) == 1 ){
                flist.push_back(f);
            }
            fclose(infile);

            Resize( (int)flist.size(), 1 );

            for(int i = 0; i < (int)flist.size(); i++){
                SetElement( i, 0, flist[i] );
            }
        }
    }


    std::cout << "WARNING! Negating Data!" << std::endl;
    for(int i = 0; i < GetElementCount()*GetDimension(); i++){
        data[i] = -data[i];
    }


    if(!load_ok){
        std::cout << "PointData: Load failed" << std::endl << std::flush;
    }
    else{
        std::cout << "PointData: " << GetVoxelCount() << " points, " << GetDim() << " dim" << std::endl << std::flush;
    }

    CalculateMinMaxMeans();
}

PointData::PointData( const PointData & data0, const PointData & data1 ) : DenseMultiDimensionalData(1,1) {
    if( data0.fname.length() > 0 ){
        fname = data0.fname;
    }
    else{
        if( data1.fname.length() > 0 ) fname = data1.fname;
    }

    if( data0.GetVoxelCount() != data1.GetVoxelCount() ){
        std::cout << "PointData: WARNING - Merging two datasets with different element counts" << std::endl << std::flush;
    }

    Resize( SCI::Max(data0.GetVoxelCount(),data1.GetVoxelCount()), data0.GetDim() + data1.GetDim() );

    std::vector<float> space( data0.GetDim() + data1.GetDim() );

    for(int i = 0; i < GetVoxelCount(); i++){
        data0.GetElement( i, &(space[0]) );
        data1.GetElement( i, &(space[data0.GetDim()]) );
        SetElement( i, &(space[0]) );
    }

    std::cout << "PointData: " << GetVoxelCount() << " points, " << GetDim() << " dim" << std::endl << std::flush;

    CalculateMinMaxMeans();

}

std::string PointData::GetFilename( ){
    return fname;
}


void PointData::Save( const char * fname ){

    FILE * outfile = fopen(fname,"wb");

    if(outfile){
        fwrite( &elemN,              sizeof(int),         1,                   outfile );
        fwrite( &dimN,               sizeof(int),         1,                   outfile );
        fwrite( &(data[0]),          sizeof(float),       GetElementCount()*GetDimension(),              outfile );
        fclose( outfile );
    }
}


void PointData::CalculateMinMaxMeans( ){
    mins.resize( GetVoxelCount(),  FLT_MAX );
    means.resize( GetVoxelCount(), 0 );
    maxs.resize( GetVoxelCount(), -FLT_MAX );
    for(int i = 0; i < GetVoxelCount(); i++){
        for(int j = 0; j < GetDim(); j++){
            float v = GetElement( i, j );
            mins[i]   = SCI::Min( mins[i], v );
            means[i] += v / (float)GetDim();
            maxs[i]   = SCI::Max( maxs[i], v );
        }
    }
    std::cout << "PointData: Data range [" << GetMinimumValue() << ", " << GetMaximumValue() << "]" << std::endl << std::flush;
}

float PointData::GetMinimumValue( int elem ) const {
    if( elem == -1 ){
        float ret = FLT_MAX;
        for(int i = 0; i < mins.size(); i++){
            ret = SCI::Min( ret, mins[i] );
        }
        return ret;
    }
    return mins[elem];
}

float PointData::GetMeanValue( int elem ) const {
    return means[elem];
}

float PointData::GetMaximumValue( int elem ) const {
    if( elem == -1 ){
        float ret = -FLT_MAX;
        for(int i = 0; i < maxs.size(); i++){
            ret = SCI::Max( ret, maxs[i] );
        }
        return ret;
    }
    return maxs[elem];
}

float PointData::GetMaximumValue( ) {
    return DenseMultiDimensionalData::GetMaximumValue();
}

float PointData::GetMinimumValue( ) {
    return DenseMultiDimensionalData::GetMinimumValue();
}




/*
PointData::PointData( const char *geom_file, const char ** data_file) : VolData( 1,1,1,1 ) {
    FILE * infile;
    infile = fopen(geom_file,"r");
    if(infile){
        float f0,f1,f2;
        while( fscanf( infile, " %f %f %f ",&f0,&f1,&f2) == 3 ){
            vertices.push_back(SCI::Vex3(f0,f1,f2));
        }
        fclose(infile);
    }

    std::vector<float> flist;
    if(data_file != 0 ){
        for(int i = 0; data_file[i] != 0; i++){
            std::cout << "loading: " << data_file[i] << std::endl; fflush(stdout);
            infile = fopen(data_file[i],"r");
            if(infile){
                float f;
                while( fscanf( infile, " %f ",&f) == 1 ){
                    flist.push_back(f);
                }
                fclose(infile);
            }
        }
    }

    __resize( vertices.size(), 1, 1, flist.size()/vertices.size() );

    for(int i = 0; i < (int)flist.size(); i++){
        SetElement( i%vertices.size(), i/vertices.size(), flist[i] );
    }

    printf("%i vertices, %i dim\n",vertices.size(),dim);
    fflush(stdout);

    mins.resize( GetVoxelCount(),  FLT_MAX );
    means.resize( GetVoxelCount(), 0 );
    maxs.resize( GetVoxelCount(), -FLT_MAX );
    for(int i = 0; i < GetVoxelCount(); i++){
        for(int j = 0; j < GetDim(); j++){
            float v = GetElement( i, j );
            mins[i]   = SCI::Min( mins[i], v );
            means[i] += v / (float)GetDim();
            maxs[i]   = SCI::Max( maxs[i], v );
        }
    }

}
*/

/*
PointData::PointData( const char ** data_file) : VolData( 1,1,1,1 ) {
    FILE * infile;

    int dsize = 0;
    std::vector<float> flist;
    if(data_file != 0 ){
        for(int i = 0; data_file[i] != 0; i++){
            std::cout << "loading: " << data_file[i] << std::endl; fflush(stdout);
            infile = fopen(data_file[i],"r");
            if(infile){
                float f;
                while( fscanf( infile, " %f ",&f) == 1 ){
                    flist.push_back(f);
                }
                fclose(infile);
            }
            if(i==0){
                dsize = flist.size();
            }
        }
    }

    __resize( dsize, 1, 1, flist.size()/dsize );

    for(int i = 0; i < (int)flist.size(); i++){
        SetElement( i%dsize, i/dsize, flist[i] );
    }

    printf("PointData: %i points, %i dim\n",dsize,dim);
    fflush(stdout);

    CalculateMinMaxMeans();

}
*/

/*
PointData::PointData( int elem, int dim ) : VolData( elem, 1, 1, dim ) { }

PointData::PointData( const char * filename, bool binary ) : VolData(1,1,1,1) {
    fname = filename;
    bool load_ok = true;

    if( binary ){
        FILE * infile = fopen( filename, "rb" );
        if(infile){
            int _sx,_dim;
            load_ok = load_ok && ( (sizeof(int)*1)        == fread( &_sx,  sizeof(int),   1,      infile ) );
            load_ok = load_ok && ( (sizeof(int)*1)        == fread( &_dim, sizeof(int),   1,      infile ) );
            __resize( _sx,1,1,_dim );
            load_ok = load_ok && ( (sizeof(float)*sx*dim) == fread( vol,   sizeof(float), sx*dim, infile ) );
            fclose( infile );

        }

    }
    else{
        std::cout << "PointData: loading " << filename << std::endl << std::flush;
        FILE * infile = fopen(filename,"r");
        if(infile){
            std::vector<float> flist;
            float f;
            while( fscanf( infile, " %f ",&f) == 1 ){
                flist.push_back(f);
            }
            fclose(infile);

            __resize( (int)flist.size(), 1, 1, 1 );

            for(int i = 0; i < (int)flist.size(); i++){
                SetElement( i, 0, flist[i] );
            }
        }
    }

    if(!load_ok){
        std::cout << "PointData: Load failed" << std::endl << std::flush;
    }
    else{
        std::cout << "PointData: " << GetVoxelCount() << " points, " << GetDim() << " dim" << std::endl << std::flush;
    }

    CalculateMinMaxMeans();
}

PointData::PointData( const PointData & data0, const PointData & data1 ) : VolData(1,1,1,1) {
    if( data0.fname.length() > 0 ){
        fname = data0.fname;
    }
    else{
        if( data1.fname.length() > 0 ) fname = data1.fname;
    }

    if( data0.GetVoxelCount() != data1.GetVoxelCount() ){
        std::cout << "PointData: WARNING - Merging two datasets with different element counts" << std::endl << std::flush;
    }

    __resize( SCI::Max(data0.GetVoxelCount(),data1.GetVoxelCount()), 1, 1, data0.GetDim() + data1.GetDim() );

    std::vector<float> space( data0.GetDim() + data1.GetDim() );

    for(int i = 0; i < GetVoxelCount(); i++){
        data0.GetElement( i, &(space[0]) );
        data1.GetElement( i, &(space[data0.GetDim()]) );
        SetElement( i, &(space[0]) );
    }

    std::cout << "PointData: " << GetVoxelCount() << " points, " << GetDim() << " dim" << std::endl << std::flush;

    CalculateMinMaxMeans();

}

std::string PointData::GetFilename( ){
    return fname;
}


void PointData::Save( const char * fname ){

    FILE * outfile = fopen(fname,"wb");

    if(outfile){
        fwrite( &sx,                sizeof(int),         1,                   outfile );
        fwrite( &dim,               sizeof(int),         1,                   outfile );
        fwrite( vol,                sizeof(float),       sx*dim,              outfile );
        fclose( outfile );
    }
}


void PointData::CalculateMinMaxMeans( ){
    mins.resize( GetVoxelCount(),  FLT_MAX );
    means.resize( GetVoxelCount(), 0 );
    maxs.resize( GetVoxelCount(), -FLT_MAX );
    for(int i = 0; i < GetVoxelCount(); i++){
        for(int j = 0; j < GetDim(); j++){
            float v = GetElement( i, j );
            mins[i]   = SCI::Min( mins[i], v );
            means[i] += v / (float)GetDim();
            maxs[i]   = SCI::Max( maxs[i], v );
        }
    }
    std::cout << "PointData: Data range [" << GetMinimumValue() << ", " << GetMaximumValue() << "]" << std::endl << std::flush;
}

float PointData::GetMinimumValue( int elem ) const {
    return mins[elem];
}

float PointData::GetMeanValue( int elem ) const {
    return means[elem];
}

float PointData::GetMaximumValue( int elem ) const {
    return maxs[elem];
}

float PointData::GetMaximumValue( ) const {
    return VolData::GetMaximumValue();
}

float PointData::GetMinimumValue( ) const {
    return VolData::GetMinimumValue();
}

*/
