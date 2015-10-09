#ifndef RENDERENGINE2D_H
#define RENDERENGINE2D_H

#include <QGLWidget>

#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/TriMesh.h>
#include <Data/PointData.h>
#include <Data/Mesh/SolidMesh.h>

#include <GL/oglFont.h>

class RenderEngine2D : public QGLWidget {

    Q_OBJECT

public:
    RenderEngine2D(QObject *parent = 0);
    
    void SetData(Data::PointData * pdata, Data::Mesh::PointMesh *pmesh, Data::Mesh::SolidMesh *tdata, SCI::Vex4 plane, SCI::Vex4 rotation);

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

    //RenderEngine render_engine;

protected:
    // Interaction based variables
    int mouse_button, mouse_x, mouse_y;

signals:
    void Updated( RenderEngine2D * );

public:
    virtual bool MouseClick(int button, int state, int x, int y);
    virtual bool MouseMotion(int button, int x, int dx, int y, int dy);
    virtual bool Keypress( int key, int x, int y );

    virtual void Draw(int width, int height);

    SCI::Vex4 rotation;
    SCI::Vex4 plane;

    Data::Mesh::ColorMesh      colormap;
    Data::Mesh::PointMesh vtmp;
    Data::Mesh::TriMesh ttmp;
    Data::MultiDimensionalDataVector * ptmp;
    Data::Mesh::EdgeMesh  iso_lines;
    Data::Mesh::PointMesh iso_verts;

    bool mouse_active;

    Data::PointData       * pdata;
    Data::Mesh::PointMesh * pmesh;
    Data::Mesh::SolidMesh * tdata;

    oglWidgets::oglFont * font;

    SCI::Vex3 pln_color;

    void UpdateMesh();
signals:
    
public slots:
    
};

#endif // RENDERENGINE2D_H
