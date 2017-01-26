#ifndef HEARTDOCK_H
#define HEARTDOCK_H

#include <QDockWidget>
#include <QSplitter>
#include <QTableWidget>

#include <muView/ParallelCoordinates.h>
#include <muView/RenderEngine.h>

#include <QT/QControlWidget.h>
#include <SCI/Camera/ThirdPersonCameraControls.h>
#include <Data/FiberDirectionData.h>
#include <Data/DistanceField.h>


class HeartDock : public QDockWidget {

    Q_OBJECT

public:
    HeartDock(SCI::ThirdPersonCameraControls * pView, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _smesh, Data::FiberDirectionData * _fdata, QWidget *parent = 0);
    
    virtual QSize minimumSizeHint() const { return QSize(  50,  50); }
    virtual QSize sizeHint()        const { return QSize( 500, 500); }

    Data::PointData          * GetPointData(){ return pdata; }
    Data::Mesh::PointMesh    * GetPointMesh(){ return pmesh; }
    Data::Mesh::SolidMesh    * GetSolidMesh(){ return tdata; }
    Data::FiberDirectionData * GetFiberData(){ return fdata; }
    Data::DistanceFieldSet   * GetDistanceField(){ return dfield; }

    void SetPointData( Data::PointData * _pdata );
    void SetDistanceFieldData( Data::DistanceFieldSet * _dfield );
    void AddImportedMesh( Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata );

public:
    static Data::Mesh::PointMesh    * OpenPointMesh( );
    static Data::Mesh::SolidMesh    * OpenSolidMesh( );
    static Data::DistanceFieldSet   * OpenDistanceField( );
    static Data::PointData          * OpenPointData( Data::PointData * pdata = 0 );
    static Data::FiberDirectionData * OpenFiberData( );

protected:
    Data::PointData          * pdata;
    Data::Mesh::PointMesh    * pmesh; 
    Data::Mesh::SolidMesh    * tdata;
    Data::FiberDirectionData * fdata;
    Data::DistanceFieldSet   * dfield;

    QSplitter           * sp_widget;
    QSplitter           * vp_widget;
    QTabWidget          * tb_widget;
    ParallelCoordinates * pr_widget;

    RenderEngine render_engine;

};

#endif // HEARTDOCK_H
