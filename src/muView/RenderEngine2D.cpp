#include <GL/glew.h>

#include <muView/RenderEngine2D.h>
#include <GL/oglCommon.h>

#include <QMouseEvent>

RenderEngine2D::RenderEngine2D(QObject *parent) : QGLWidget() {

    ptmp = 0;
    ptmp = new Data::MultiDimensionalDataVector( 0, 0 );
    mouse_active = false;

    pdata = 0;
    pmesh = 0;
    tdata = 0;

    mouse_button = Qt::NoButton;
    mouse_x = 0;
    mouse_y = 0;

    font = 0;

    setMouseTracking(true);

    pln_color.Set(1,0,0);

}


void RenderEngine2D::initializeGL(){
    glewInit( );
}

void RenderEngine2D::paintGL(){

    // Clear the display
    glClearColor(1,1,1,1);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    Draw( size().width(), size().height() );

}

void RenderEngine2D::mouseDoubleClickEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonDblClick, x, y );

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void RenderEngine2D::mouseMoveEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseMotion( mouse_button, x, x-mouse_x, y, y-mouse_y );

    mouse_x = x;
    mouse_y = y;

}

void RenderEngine2D::mousePressEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonPress, x, y );

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void RenderEngine2D::mouseReleaseEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonRelease, x, y );

    mouse_button = Qt::NoButton;
    mouse_x = x;
    mouse_y = y;

}

void RenderEngine2D::keyPressEvent ( QKeyEvent * event ){

    Keypress( event->key(), mouse_x, mouse_y );

}

void RenderEngine2D::keyReleaseEvent ( QKeyEvent * event ){
    update();
}


void RenderEngine2D::SetData(Data::PointData * _pdata, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata, SCI::Vex4 pln, SCI::Vex4 rot){
    // set new plane and rotation angle
    plane = pln;
    rotation = rot;

    pdata = _pdata;
    pmesh = _pmesh;
    tdata = _tdata;

    UpdateMesh();
}

void RenderEngine2D::UpdateMesh(){
    if( pdata == 0 || pmesh == 0 || tdata == 0 ) return;

    // clear old data
    ptmp->Resize( 0, pdata->GetDimension() );
    vtmp.Clear();
    ttmp.clear();

    // extract new plane
    tdata->ExtractPlane( plane, *pmesh, *pdata, ttmp, vtmp, *ptmp );

    emit Updated( this );
}

void RenderEngine2D::Draw(int width, int height){
    glViewport( 0,0,width,height );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    float right = 1.0f;
    float top   = 1.0f;
    if( width >= height ){
        right = (float)width/(float)height;
    }
    else{
        top = (float)height/(float)width;
    }
    glScalef( 1.0f/right, 1.0f/top, 1.0f );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glEnable(GL_DEPTH_TEST);

    glLineWidth(12.0f);
    glColor3fv(pln_color.data);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-1*right,-1*top,0);
    glVertex3f(-1*right, 1*top,0);
    glVertex3f( 1*right, 1*top,0);
    glVertex3f( 1*right,-1*top,0);
    glEnd();
    /*
    glPushMatrix();
        glTranslatef( -right+0.14f, top-0.12f, 0.0f );
        glScalef(0.1f,0.1f,0.1f);
        glRotatef( rotation.x, rotation.y, rotation.z, rotation.w );
        glLineWidth( 7.0f );
        glBegin(GL_LINES);
            glColor3f(1,0,0); glVertex3f(-1,-1,-1); glVertex3f( 1,-1,-1);
            glColor3f(0,1,0); glVertex3f(-1,-1,-1); glVertex3f(-1, 1,-1);
            glColor3f(0,0,1); glVertex3f(-1,-1,-1); glVertex3f(-1,-1, 1);
        glEnd();
        if( font ){
            glColor3f(0,0,0);
            glPushMatrix();
                glRotatef( 0, 1, 0, 0 );
                glScalef(1,1,0.001f);
                glTranslatef(0.5f,-0.8f,0);
                font->Print( "x" );
                glTranslatef(-1.8f,1.3f,0);
                font->Print( "y" );
            glPopMatrix();
            glPushMatrix();
                glRotatef( 90, 1, 0, 0 );
                glScalef(1,1,0.001f);
                glTranslatef(0.5f,-0.8f,0);
                font->Print( "x" );
                glTranslatef(-1.8f,1.3f,0);
                font->Print( "z" );
            glPopMatrix();
            glPushMatrix();
                glRotatef( 90, 0, -1, 0 );
                glScalef(1,1,0.001f);
                glTranslatef(0.5f,-0.8f,0);
                font->Print( "z" );
                glTranslatef(-1.8f,1.3f,0);
                font->Print( "y" );
            glPopMatrix();
        }
    glPopMatrix();
    */

    SCI::Vex3 c = vtmp.bb.GetCenter();
    float     s = vtmp.bb.GetMaximumDimensionSize();

    glPushMatrix();
        glScalef( 1.9f/s, 1.9f/s, 1.0f );
        glRotatef( rotation.x, rotation.y, rotation.z, rotation.w );
        glTranslatef( -c.x, -c.y, -c.z );

        glDisable(GL_DEPTH_TEST);
        glLineWidth(2.0f);
        ttmp.Draw( vtmp, colormap );
        //ttmp.Draw( vtmp, SCI::Vex4(0,0,0,1) );
        iso_lines.Draw( iso_verts, SCI::Vex4(0,0,0,1) );
    glPopMatrix();
}

bool RenderEngine2D::MouseClick(int button, int state, int x, int y){
    //pView->Save("view.txt");
    mouse_active = ( state == QMouseEvent::MouseButtonPress );
    return mouse_active;
}


bool RenderEngine2D::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if(button == Qt::LeftButton)  { }//pView->Rotate(-(float)(dx),(float)(dy));
        if(button == Qt::MiddleButton){ }//pView->Translate((float)(dx),(float)(dy));
        if(button == Qt::RightButton){  plane.w += dy/10.0f; UpdateMesh(); update(); }
    }
    return mouse_active;
}

bool RenderEngine2D::Keypress( int key, int x, int y ){
    return false;
}
