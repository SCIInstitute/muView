#ifndef PCAVIEW_H
#define PCAVIEW_H

#include <QGLWidget>


#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/TriMesh.h>
#include <Data/PointData.h>
#include <Data/Mesh/SolidMesh.h>

#include <DimensionalityReduction.h>

#include <GL/oglFont.h>

class PCAView : public QGLWidget
{
    Q_OBJECT
public:
    PCAView(QObject *parent = 0);
    
    void SetData(Data::PointData * pdata, Data::Mesh::PointMesh *pmesh, Data::Mesh::SolidMesh *tdata, Data::Mesh::ColorMesh     * colormap );

    virtual QSize minimumSizeHint() const { return QSize(  50,  50); }
    virtual QSize sizeHint()        const { return QSize( 500, 500); }

protected:

    virtual void initializeGL();
    virtual void paintGL();

public:
    virtual void mouseDoubleClickEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void keyReleaseEvent ( QKeyEvent * event );
    DimensionalityReduction* getDMR();

protected:
    // Interaction based variables
    int mouse_button, mouse_x, mouse_y;

public:
    virtual bool MouseClick(int button, int state, int x, int y);
    virtual bool MouseMotion(int button, int x, int dx, int y, int dy);
    virtual bool Keypress( int key, int x, int y );

    virtual void Draw(int width, int height);

    bool mouse_active;

    Data::PointData       * pdata;
    Data::Mesh::PointMesh * pmesh;
    Data::Mesh::SolidMesh * tdata;
    Data::Mesh::ColorMesh * colormap;

    DimensionalityReduction dmr;
    int dim0, dim1;

    Data::DenseMultiDimensionalData pca_out;
    SCI::Subset full_set;
    SCI::Subset features;

    SCI::BoundingBox bb;

    SCI::Vex4 paint_color;
    bool painting;

    std::vector<SCI::Vex2> paint_points;

    oglWidgets::oglFont * font;


signals:

public slots:
    void selectPaintColor();
    void EnablePainting();
    void DisablePainting();
    void ModifyPCADim0( int dim );
    void ModifyPCADim1( int dim );


};

#endif // PCAVIEW_H
