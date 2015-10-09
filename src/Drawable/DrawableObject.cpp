#include <Drawable/DrawableObject.h>
#include <Qt>

#include <GL/oglCommon.h>

using namespace Drawable;

const int DrawableObject::NoButton       = Qt::NoButton;
const int DrawableObject::LeftButton     = Qt::LeftButton;
const int DrawableObject::RightButton    = Qt::RightButton;
const int DrawableObject::MiddleButton   = Qt::MiddleButton;
const int DrawableObject::XButton1       = Qt::XButton1;
const int DrawableObject::XButton2       = Qt::XButton2;

const int DrawableObject::Down           = 0;
const int DrawableObject::Up             = 1;
const int DrawableObject::Double         = 2;

const int DrawableObject::Key_Delete     = 0x1000007;
const int DrawableObject::Key_Enter      = 0x1000004;
const int DrawableObject::Key_Backspace  = 0x1000003;



DrawableObject::DrawableObject(){

}

void DrawableObject::SetShape( int _u0, int _v0, int w, int h ){
    u0 = _u0;
    v0 = _v0;
    width = w;
    height = h;
}

bool DrawableObject::MouseClick(int button, int state, int x, int y){
    return false;
}

bool DrawableObject::MouseMotion(int button, int x, int dx, int y, int dy){
    return false;
}

bool DrawableObject::Keypress( int key, int x, int y ){
    return false;
}

bool DrawableObject::ClearSelection( ){
    return true;
}

bool DrawableObject::isPointInFrame( int x, int y ){
    return (x>=u0 && y>=v0 && x<=(u0+width) && y<=v0+height);
}

void DrawableObject::SetViewport(){
    glViewport( u0,v0,width,height);
}

void DrawableObject::Clear( ){
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    // Paint the area white (i.e. erase any pixel data)
    glColor3f(1,1,1);
    glBegin(GL_QUADS);
        glVertex3f(-1,-1,-1);
        glVertex3f( 1,-1,-1);
        glVertex3f( 1, 1,-1);
        glVertex3f(-1, 1,-1);
    glEnd();
}

void DrawableObject::DrawFrame(){
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    // Draw the red frame around the vis
    glLineWidth(4.0f);
    glColor3f(1,0,0);
    glBegin(GL_LINE_LOOP);
        glVertex3f(-1,-1,0);
        glVertex3f( 1,-1,0);
        glVertex3f( 1, 1,0);
        glVertex3f(-1, 1,0);
    glEnd();

}
