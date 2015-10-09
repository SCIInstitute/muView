#include <QApplication>

#include <iostream>
#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/SolidMesh.h>
#include <Data/Mesh/TetMesh.h>
#include <Data/Mesh/HexMesh.h>
#include <Data/DistanceField.h>


void df_create(char **argv, int argc );
void df_merge(char **argv, int argc );

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QCoreApplication::setOrganizationName("SCI Institute");
    QCoreApplication::setOrganizationDomain("Uncertainty");
    QCoreApplication::setApplicationName("dfield");

    for(int i = 0; i < argc; i++){
        std::cout << i << ": " << argv[i] << std::endl << std::flush;
    }

    if( argc >= 2 && strncmp( argv[1], "-create", 2 ) == 0 ){
        df_create( argv, argc );
    }
    if( argc >= 2 && strncmp( argv[1], "-merge", 2 ) == 0 ){
        df_merge( argv, argc );
    }

    return 0;

}


void df_merge(char **argv, int argc){

    std::string outfile;
    Data::DistanceFieldSet dfs;
    for(int i = 0; i < argc; i++){
        if( strncmp( argv[i], "-output", 2 ) == 0 ){
            i++;
            outfile = argv[i];
        }
        else if( QString(argv[i]).endsWith(".dfield") || QString(argv[i]).endsWith(".df") ){
            dfs.Load( argv[i] );
        }
    }

    if( outfile.size() > 0 ){
        dfs.Save( outfile.c_str() );
    }

}


void df_create(  char ** argv, int argc ){

    Data::Mesh::PointMesh * p_mesh = 0;
    Data::Mesh::SolidMesh * s_mesh = 0;
    Data::PointData       * p_data = 0;

    QString outfile;

    bool  smin   = true;
    bool  smax   = false;
    float isoval = 0;

    for(int i = 0; i < argc; i++){
        QString fname(argv[i]);

        if( fname.compare( "-isoval" ) == 0 || fname.compare( "-iso_val" ) == 0 ){
            isoval = (float)atof( argv[i+1] );
            i++;
        }
        else if( fname.compare( "-min" ) == 0 ){
            smin = true; smax = false;
        }
        else if( fname.compare( "-max" ) == 0 ){
            smax = true; smin = false;
        }
        else if( fname.endsWith(".point") ){ // Loading a point data file
            p_mesh = new Data::Mesh::PointMesh( fname.toLocal8Bit().data(), true, false );
        }
        else if( fname.endsWith(".pts") ){ // Loading raw pts data file
            p_mesh = new Data::Mesh::PointMesh( fname.toLocal8Bit().data(), false, false );
        }
        else if( fname.endsWith(".btet") || fname.endsWith(".tet") ){
            s_mesh = new Data::Mesh::TetMesh( fname.toLocal8Bit().data() );
        }
        else if( fname.endsWith(".bhex") || fname.endsWith(".hex") ){
            s_mesh = new Data::Mesh::HexMesh( fname.toLocal8Bit().data() );
        }
        else if( fname.endsWith(".pdata") || fname.endsWith(".sol") || fname.endsWith(".sol") ){
            Data::PointData * ptmp0 = p_data;
            Data::PointData * ptmp1 = new Data::PointData( fname.toLocal8Bit().data(), fname.endsWith(".pdata") );
            if(p_data == 0) {
                p_data = ptmp1;
            }
            else {
                p_data = new Data::PointData( *ptmp0, *ptmp1 );
                delete ptmp0;
                delete ptmp1;
            }
        }
        else if( fname.endsWith(".dfield") || fname.endsWith(".df") ){
            outfile = fname;
        }
    }

    if( p_mesh != 0 && s_mesh != 0 && p_data != 0 ){
        Data::DistanceFieldSet dfs;
        dfs.Insert( smin, smax, isoval )->Process( p_mesh, s_mesh, p_data );
        dfs.Save( outfile.toLocal8Bit().data() );
    }

}
