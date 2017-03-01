#ifndef CHARTCLOUD_H
#define CHARTCLOUD_H

#include <QObject>
#include <QImage>
#include <QTimer>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChart>

using namespace QtCharts;

#include <Data/VolumetricData.h>
#include <Data/PointData.h>
#include <Data/MappedData.h>
#include <Data/Mesh/TetMesh.h>
#include <Data/Mesh/HexMesh.h>
#include <Data/Mesh/SolidMesh.h>
#include <Data/Mesh/EdgeMesh.h>
#include <Data/Mesh/OrientationMesh.h>
#include <Data/FiberDirectionData.h>
#include <Data/KMeansClustering.h>

#include <muView/VolumeRenderer.h>
#include <muView/VoxelAssociativity.h>
#include <muView/Histogram.h>
#include <muView/ParallelCoordinates.h>
#include <muView/ChartRect.h>

#include <SCI/Camera/FrustumProjection.h>
#include <SCI/Camera/OrthoProjection.h>
#include <SCI/Camera/ThirdPersonCameraControls.h>

class ChartCloud : public QGLWidget {

    Q_OBJECT


protected:

    void UpdateView();

    bool need_view_update;

public:
    virtual QSize minimumSizeHint() const { return QSize(  50,  50); }
    virtual QSize sizeHint()        const { return QSize( 500, 500); }

protected:

    virtual void initializeGL();
 //   virtual void paintGL();
    virtual void paintEvent(QPaintEvent *event);
    virtual void showEvent(QShowEvent *event);

public:
    virtual void mouseDoubleClickEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void keyReleaseEvent ( QKeyEvent * event );

    //RenderEngine render_engine;

protected:
    // Interaction based variables
    int mouse_button, mouse_x, mouse_y;


public:
    ChartCloud(QObject *parent = 0);
    
    // Draw the data
    virtual void Draw(int width, int height);

    // Sets the data
    void Initialize( );

    virtual bool MouseClick(int button, int state, int x, int y);
    virtual bool MouseMotion(int button, int x, int dx, int y, int dy);
    virtual bool Keypress( int key, int x, int y );

    inline void NeedUpdate(){ need_view_update = true; }


    void RecalculateSil();

    std::vector<Data::Mesh::PointMesh*>       addl_points;
    std::vector<Data::Mesh::SolidMesh*>       addl_solid;
    std::vector<Data::Mesh::OrientationMesh*> addl_sil_mesh;
    std::vector<Data::Mesh::ColorMesh*>       addl_sil_color;
    std::vector< std::vector<int>* >          addl_mesh_ro;


    Data::Mesh::OrientationMesh sil_mesh;
    Data::Mesh::ColorMesh       sil_color;

    bool cluster_histogram;


    ParallelCoordinates * parallel_coordinates;

    Data::PointData          * pdata;
    Data::Mesh::PointMesh    * pmesh;
    Data::Mesh::SolidMesh    * tdata;

    SCI::Mat4                          tform;

    bool mouse_active;

    Data::Mesh::ColorMesh     * colormap;

    SCI::SequentialColormap  * seq_cmap;
    SCI::CatagoricalColormap * cat_cmap;

    //VoxelAssociativity *vox_assoc;

    Data::Mesh::PointMesh *iso_points;
    Data::Mesh::TriMesh   *iso_tris;
    Data::Mesh::TetMesh   *iso_tets;
    Data::Mesh::HexMesh   *iso_hexs;
    Data::Mesh::ColorMesh *iso_color;

    Data::Mesh::PointMesh *df_points;
    Data::Mesh::TriMesh   *df_tris;
    Data::Mesh::ColorMesh *df_color;

    //Data::DenseMultiDimensionalData *edge_data;
    //Data::Mesh::EdgeMesh            *edge_mesh;

    SCI::Vex4 * pln[3];
    SCI::Vex3 * pln_color[3];

    KMeansClustering * cluster;

    std::vector<int> iso_tris_ro;
    std::vector<int> df_tris_ro;
    std::vector<int> tri_mesh_ro;
    //std::vector<int> edge_mesh_ro;

    int  draw_mode;
    int  color_mode;
    oglWidgets::oglFont * font;


    SCI::FrustumProjection           proj;
    SCI::ThirdPersonCameraControls * pView;

    double clpX[4];
    double clpY[4];
    double clpZ[4];
    bool useClipX, useClipY, useClipZ;
















    std::vector<QChartView *> allChartViews;

    void createChartRects(int number);
    void setupViewport(int width, int height);
    void resizeGL(int width, int height);

    QList<ChartRect *> chartRects;
};

#endif // CHARTCLOUD_H
