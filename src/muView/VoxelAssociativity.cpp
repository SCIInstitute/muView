#include <muView/VoxelAssociativity.h>

#include <iostream>


VoxelAssociativityThread::VoxelAssociativityThread(QObject *parent) : QThread(parent){
    pmesh = 0;
    tdata = 0;
    vol_render = 0;
    assoc = 0;
    vox_start = 0;
    vox_count = 0;
}

VoxelAssociativityThread::~VoxelAssociativityThread(){
    terminate();
}

void VoxelAssociativityThread::run(){
    int dim = vol_render->GetDim();

    for(int i = vox_start; i < (vox_start+vox_count); i++ ){
        int x = i%dim;
        int y = (i/dim)%dim;
        int z = (i/dim/dim);
        assoc->at( i ) = tdata->ContainingElement( vol_render->VoxelCenter( x, y, z ), *pmesh );
    }
}

void VoxelAssociativityThread::Launch( VolumeRenderer *_vol_render, Data::Mesh::PointMesh *_pmesh, Data::Mesh::SolidMesh *_tdata, std::vector<int> * _assoc, int vox_id, int cnt  ){
    terminate();

    pmesh = _pmesh;
    tdata = _tdata;
    vol_render = _vol_render;
    assoc = _assoc;
    vox_start = vox_id;
    vox_count = cnt;

    start();
}



VoxelAssociativity::VoxelAssociativity() {
    pmesh = 0;
    tdata = 0;
    vol_render = 0;
}

VoxelAssociativity::~VoxelAssociativity(){ }

void VoxelAssociativity::Launch( VolumeRenderer *_vol_render, Data::Mesh::PointMesh *_pmesh, Data::Mesh::SolidMesh *_tdata ){

    if( pmesh != _pmesh || tdata != _tdata ){
        threads[0].terminate();
        threads[1].terminate();
        threads[2].terminate();
        threads[3].terminate();

        pmesh = _pmesh;
        tdata = _tdata;
        vol_render = _vol_render;

        assoc.clear();

        if(vol_render != 0 && pmesh != 0 && tdata != 0){
            int dim = vol_render->GetDim();
            assoc.resize( dim*dim*dim, -1 );

            int cnt = dim*dim*dim / 4;
            threads[0].Launch( vol_render, pmesh, tdata, &assoc, cnt*0, cnt );
            threads[1].Launch( vol_render, pmesh, tdata, &assoc, cnt*1, cnt );
            threads[2].Launch( vol_render, pmesh, tdata, &assoc, cnt*2, cnt );
            threads[3].Launch( vol_render, pmesh, tdata, &assoc, cnt*3, cnt );
        }
    }
}

bool VoxelAssociativity::Wait( ){
    if(! threads[0].wait( ) ) return false;
    if(! threads[1].wait( ) ) return false;
    if(! threads[2].wait( ) ) return false;
    if(! threads[3].wait( ) ) return false;
    return true;
}



/*

VoxelAssociativity::VoxelAssociativity(QObject *parent) : QThread(parent){
    pmesh = 0;
    tdata = 0;
    vol_render = 0;
}

VoxelAssociativity::~VoxelAssociativity(){
    terminate();
}

void VoxelAssociativity::run(){
    if(vol_render == 0 || pmesh == 0 || tdata == 0) return;
    if( assoc.size() != 0 ) return;

    int dim = vol_render->GetDim();
    assoc.resize( dim*dim*dim, -1 );
    std::cout << vol_render->GetDim() << std::endl << std::flush;

    #pragma omp parallel
    for(int z = 0; z < vol_render->GetDim(); z++){
        for(int y = 0; y < vol_render->GetDim(); y++){
            for(int x = 0; x < vol_render->GetDim(); x++){
                assoc[ vol_render->GetVoxelID(x,y,z) ] = tdata->ContainingElement( vol_render->VoxelCenter( x, y, z ), *pmesh );
            }
        }
    }
}

void VoxelAssociativity::Launch( VolumeRenderer *_vol_render, Data::Mesh::PointMesh *_pmesh, Data::Mesh::SolidMesh *_tdata ){

    if( pmesh != _pmesh || tdata != _tdata ){
        terminate();

        pmesh = _pmesh;
        tdata = _tdata;
        vol_render = _vol_render;

        assoc.clear();

        start();
    }
}
*/

