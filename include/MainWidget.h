
#ifndef GLWIDGET_H
#define GLWIDGET_H

#ifdef WIN32
    #include <QtOpenGL/QGLWidget>
#else
    #include <QGLWidget>
#endif
#include <QMenuBar>
#include <QtGui>

#include <Data/NrrdData.h>
#include <Data/VolData.h>
#include <Data/MeshData.h>
#include <Data/PointData.h>
#include <Data/MappedData.h>

#include <Dialog/Metadata.h>

#include <Drawable/DimRedDisplay.h>
#include <Drawable/VolumeDisplay.h>
#include <Drawable/PointDisplay.h>
#include <Drawable/Provenance.h>
#include <Drawable/BarChart.h>
#include <Drawable/MeshDisplay.h>

#include <SCI/Timer.h>
#include <SCI/Subset.h>
#include <SCI/Vex3.h>

#include <time.h>
#include <DimensionalityReduction.h>


class MainWidget : public QGLWidget
{
    Q_OBJECT

public:

    MainWidget( Dialog::Metadata * _data, QMenuBar * _menu_bar, QWidget * parent = 0);
    ~MainWidget();

    virtual QSize minimumSizeHint() const;
    virtual QSize sizeHint() const;

    void AttachMenus( );
    void DetachMenus( );

protected:

    virtual void showEvent ( QShowEvent * event );
    virtual void hideEvent ( QHideEvent * event );

    virtual void initializeGL();
    virtual void paintGL();

public:
    virtual void mouseDoubleClickEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void keyReleaseEvent ( QKeyEvent * event );

protected:
    // Variables related to the detachable menu system
    // This whole detachable menu thing could be better implemented
    QMenuBar                                * menu_bar;
    QMenu                                   * dr_method_menu;
    QAction                                 * dr_method_menu_action;
    QActionGroup                            * dr_method_group;

    QMenu                                   * view_menu;
    QAction                                 * view_reset;
    QAction                                 * view_menu_action;

protected:
    // Interaction based variables
    int mouse_button, mouse_x, mouse_y;

protected:
    // Containers for data
    Data::ProxyData          * data;

    // Tools used to draw data
    Drawable::DimRedDisplay    draw_dr;
    Drawable::DrawableObject * draw_data;
    Drawable::Provenance       prov;


};


#endif
