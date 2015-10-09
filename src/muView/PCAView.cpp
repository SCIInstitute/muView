#include <GL/glew.h>

#include <muView/PCAView.h>
#include <GL/oglCommon.h>

#include <QMouseEvent>

PCAView::PCAView(QObject *parent) : QGLWidget() {

    mouse_active = false;

    pdata = 0;
    pmesh = 0;
    tdata = 0;

    mouse_button = Qt::NoButton;
    mouse_x = 0;
    mouse_y = 0;

    setMouseTracking(true);

    painting = false;

    dim0 = 0;
    dim1 = 1;

    font = 0;

}


void PCAView::initializeGL(){
    glewInit( );
}

void PCAView::paintGL(){

    glViewport( 0,0,size().width(), size().height() );

    // Clear the display
    glClearColor(1,1,1,1);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    Draw( size().width(), size().height() );

}


void PCAView::Draw(int width, int height){

    SCI::Vex3 pnt(0,0,0);
    std::vector<float> tmp( pca_out.GetDimension() );

    bb.Clear();
    for(int i = 0; i < dmr.GetElementCount(); i++){
        dmr.GetElementPoint( i, tmp.data() );
        bb.Expand( SCI::Vex3( tmp[dim0], tmp[dim1], 0 ) );
    }

    glEnable(GL_DEPTH_TEST);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glScalef( 1.9f/bb.GetSize().x, 1.9f/bb.GetSize().y, 1.0f );
    glTranslatef( -bb.GetCenter().x, -bb.GetCenter().y, 0 );

    glPointSize(2.0f);
    glBegin( GL_POINTS );
    for(int i = 0; i < dmr.GetElementCount(); i++){
        glColor4ubv( (GLubyte*) &(colormap->at(i)) );
        //dmr.GetElementPoint( i, pnt.data );
        //glVertex3fv( pnt.data );
        dmr.GetElementPoint( i, tmp.data() );
        glVertex3f( tmp[dim0], tmp[dim1], 0 );
    }
    glEnd();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glColor4f(paint_color.r, paint_color.g, paint_color.b, 0.4f );
    glPointSize(10.0f);
    glLineWidth(10.0f);

    glBegin(GL_POINTS);
    for(int i = 0; i < paint_points.size(); i++){
        glVertex3f( paint_points[i].x, paint_points[i].y, -0.1f );
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < paint_points.size(); i++){
        glVertex3f( paint_points[i].x, paint_points[i].y, -0.1f );
    }
    glEnd();

    glDisable(GL_BLEND);

    if(font){
        std::stringstream title0;
        if( dim0 == 0 ){ title0 << "1st Principal Component"; }
        if( dim0 == 1 ){ title0 << "2nd Principal Component"; }
        if( dim0 == 2 ){ title0 << "3rd Principal Component"; }
        if( dim0 >= 3 ){ title0 << dim0 << "th Principal Component"; }

        std::stringstream title1;
        if( dim1 == 0 ){ title1 << "1st Principal Component"; }
        if( dim1 == 1 ){ title1 << "2nd Principal Component"; }
        if( dim1 == 2 ){ title1 << "3rd Principal Component"; }
        if( dim1 >= 3 ){ title1 << dim1 << "th Principal Component"; }

        glColor3f(0,0,0);
        float aspect = (float)width/(float)height;
        glPushMatrix();
            glTranslatef(-0.90f,-0.90f,0);
            glScalef(0.07/aspect,0.07,0);
            font->Print( title0.str().c_str() );
        glPopMatrix();

        glPushMatrix();
            glTranslatef(-0.93f,-0.86f,0);
            glRotatef(90,0,0,1);
            glScalef(0.07,0.07/aspect,0);
            font->Print( title1.str().c_str() );
        glPopMatrix();

        float mscl = SCI::Max( bb.GetSize().x, bb.GetSize().y );
        float l1 = 0.7f * bb.GetSize().x / mscl;
        float l2 = 0.7f * bb.GetSize().y / mscl * aspect;
        glBegin(GL_LINES);
        glVertex3f(-0.98f,   -0.95f,   0);
        glVertex3f(-0.98f+l1,-0.95f,   0);
        glVertex3f(-0.98f,   -0.95f,   0);
        glVertex3f(-0.98f,   -0.95f+l2,0);
        glEnd();
        bb.GetSize().Print();

    }

}

void PCAView::mouseDoubleClickEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonDblClick, x, y );

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void PCAView::mouseMoveEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseMotion( mouse_button, x, x-mouse_x, y, y-mouse_y );

    mouse_x = x;
    mouse_y = y;

}

void PCAView::mousePressEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonPress, x, y );

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void PCAView::mouseReleaseEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonRelease, x, y );

    mouse_button = Qt::NoButton;
    mouse_x = x;
    mouse_y = y;

}

void PCAView::keyPressEvent ( QKeyEvent * event ){

    Keypress( event->key(), mouse_x, mouse_y );

}

void PCAView::keyReleaseEvent ( QKeyEvent * event ){
    update();
}


void PCAView::SetData(Data::PointData * _pdata, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata, Data::Mesh::ColorMesh * _colormap){

    pdata    = _pdata;
    pmesh    = _pmesh;
    tdata    = _tdata;
    colormap = _colormap;

    std::cout << "PCAView: " << dmr.GetMethodCount() << " methods" << std::endl;
    for(int i = 0; i < dmr.GetMethodCount(); i++){
        std::cout << "PCAView: " << dmr.GetMethodName(i) << std::endl;
    }

    //pdata->GetElementCount();

    full_set = SCI::Subset( pdata->GetElementCount() );
    features.GetRandomSubset(full_set, 1000);

    pca_out.Resize( pdata->GetElementCount(), 3 );

    std::cout << "PCAView: Starting PCA with " << features.size() << " features" << std::endl;
    dmr.Start(0, *pdata, features, pca_out, full_set );
    std::cout << "PCAView: Finished PCA feature phase" << std::endl;

    bb.Clear();

}

#include <SCI/Geometry/CompGeom.h>

bool PCAView::MouseClick(int button, int state, int x, int y){
    //pView->Save("view.txt");

    if( painting && state == QMouseEvent::MouseButtonRelease ){

        SCI::Vex2 scl = SCI::Vex2( 1.9f/bb.GetSize().x, 1.9f/bb.GetSize().y );
        SCI::Vex2 off = SCI::Vex2( -bb.GetCenter().x, -bb.GetCenter().y );

        float tmp[3];
        SCI::Vex2 pnt;
        for(int i = 0; i < dmr.GetElementCount(); i++){
            dmr.GetElementPoint( i, tmp );
            pnt.Set( tmp[dim0], tmp[dim1] );
            pnt = (pnt + off) * scl;
            for(int j = 1; j < paint_points.size(); j++){
                if( PointSegmentDistance( paint_points[j-1], paint_points[j], pnt ) < 0.1f ){
                    colormap->at(i) = paint_color.UIntColor();
                    break;
                }
            }
        }
        paint_points.clear();
        //draw_count = 0;
        update();
    }

    mouse_active = ( state == QMouseEvent::MouseButtonPress );
    return mouse_active;
}


bool PCAView::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if(button == Qt::LeftButton)  {
            if( painting ){
                paint_points.push_back( SCI::Vex2((float)x/(float)size().width(),(float)y/(float)size().height())*2.0f-1.0f );
                update();
            }
        }
        if(button == Qt::MiddleButton){ }
        if(button == Qt::RightButton){ }
    }
    return mouse_active;
}

bool PCAView::Keypress( int key, int x, int y ){
    return false;
}

#include <QColorDialog>

void PCAView::selectPaintColor(){
    SCI::CatagoricalColormap cm;
    cm.LoadDefaultMap();

    SCI::Vex4 init_col = cm.GetColor( rand() );

    QColor color = QColorDialog::getColor( QColor( (int)(init_col.r*255.0f), (int)(init_col.g*255.0f), (int)(init_col.b*255.0f), 50 ), this);
    if (color.isValid()) {
        paint_color.Set( color.redF(), color.greenF(), color.blueF(), color.alphaF() );
    }
}

void PCAView::EnablePainting(){
    painting = true;
}

void PCAView::DisablePainting(){
    painting = false;
}

void PCAView::ModifyPCADim0( int dim ){
    if( dim < 0 ){ dim0 = 0; return; }
    if( dim >= dmr.GetLowDimension() ){ dim0 = dmr.GetLowDimension()-1; return; }
    dim0 = dim;
    update();
}

void PCAView::ModifyPCADim1( int dim ){
    if( dim < 0 ){ dim1 = 0; return; }
    if( dim >= dmr.GetLowDimension() ){ dim1 = dmr.GetLowDimension()-1; return; }
    dim1 = dim;
    update();
}

