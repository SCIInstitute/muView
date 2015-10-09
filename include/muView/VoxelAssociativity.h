#ifndef VOXELASSOCIATIVITY_H
#define VOXELASSOCIATIVITY_H

#include <QThread>

#include <map>
#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/SolidMesh.h>
#include <muView/VolumeRenderer.h>


class VoxelAssociativityThread : public QThread {

    Q_OBJECT

public:
    VoxelAssociativityThread(QObject *parent = 0);
    ~VoxelAssociativityThread();

    virtual void run();

    void Launch( VolumeRenderer * _vol_render, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata, std::vector<int> * _assoc, int vox_id, int cnt );

public:

    VolumeRenderer  * vol_render;
    Data::Mesh::PointMesh * pmesh;
    Data::Mesh::SolidMesh * tdata;

    std::vector<int> * assoc;
    int vox_start, vox_count;


signals:

public slots:

};

class VoxelAssociativity {

public:
    VoxelAssociativity();
    ~VoxelAssociativity();

    void Launch( VolumeRenderer * _vol_render, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata );

    bool Wait( );

public:

    VolumeRenderer        * vol_render;
    Data::Mesh::PointMesh * pmesh;
    Data::Mesh::SolidMesh * tdata;

    std::vector<int> assoc;

    VoxelAssociativityThread threads[4];

};

#endif // VOXELASSOCIATIVITY_H
