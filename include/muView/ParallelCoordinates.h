#ifndef PARALLELCOORDINATES_H
#define PARALLELCOORDINATES_H

#include <QGLWidget>

#include <Data/PointData.h>
#include <Data/Mesh/PointMesh.h>

class ParallelCoordinates : public QGLWidget {

    Q_OBJECT

public:
    ParallelCoordinates(QWidget *parent);

    void SetData( Data::MultiDimensionalData * _pdata, Data::Mesh::ColorMesh * _cm );

    virtual QSize minimumSizeHint() const { return QSize( 50,  50); }
    virtual QSize sizeHint()        const { return QSize(500, 200); }

public slots:

    void Reset();

protected:

    virtual void resizeGL ( int width, int height );
    virtual void initializeGL();
    virtual void paintGL();

    virtual void mousePressEvent ( QMouseEvent * event ) ;
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * event );

protected:
    Data::MultiDimensionalData * pdata;
    Data::Mesh::ColorMesh      * color;

    int curDraw;
    std::vector< std::pair<float,int> > dimLoc;
    std::vector< float >                divLoc;

    float minv;
    float maxv;
    int   dim;

    int selected;

};

#endif // PARALLELCOORDINATES_H
