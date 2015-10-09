//#include <GL/glew.h>

#include <Drawable/MeshDisplay.h>
#include <Data/MeshData.h>
#include <GL/oglCommon.h>

using namespace Drawable;

MeshDisplay::MeshDisplay(){
    proj.Set( 40.0f,40.0f, 0.1f,100.0f);
    view.Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    view.Load("view.txt");
    data         = 0;
    vol_data     = 0;
    curr_node    = 0;
    updated      = false;
    mouse_active = false;
}

void MeshDisplay::Initialize( Data::ProxyData & _data, Drawable::Provenance & _prov ){
    data     = &_data;
    prov     = &_prov;
    //std::vector<SCI::Vex3> * vert = &((Data::MeshData*)data)->GetVertices();
    std::vector<SCI::Vex3> * vert = &data->vert;

    col_data = new unsigned int[ data->GetX()*data->GetY()*data->GetZ() ];

    bbmin = SCI::VEX3_MAX;
    bbmax = SCI::VEX3_MIN;

    for(int i = 0; i < (int)vert->size(); i++){
        bbmin = SCI::Min( bbmin, vert->at(i) );
        bbmax = SCI::Max( bbmax, vert->at(i) );
    }
}

unsigned int * MeshDisplay::ClearColor(){
    return (unsigned int*)memset( col_data, 0, sizeof(int)*data->GetX()*data->GetY()*data->GetZ() );
}

void MeshDisplay::Draw(){
    if(data == 0) return;

    if( (prov->GetCurrentNode() != curr_node) || (!updated && curr_node->isDone()) ){
        curr_node = prov->GetCurrentNode();
        updated   = curr_node->isDone();
        curr_node->ColorData( ClearColor( ) );
    }

    glViewport( u0,v0,width,height);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMultMatrixf( proj.GetMatrix().data );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glMultMatrixf( view.GetView().data );

    SCI::Vex3 minmax[2];
    minmax[0] = SCI::Vex3(-1.0,-1.0,-1.0 );
    minmax[1] = SCI::Vex3( 1.0, 1.0,1.0f );

    SCI::Mat4 mvp  = proj.GetMatrix() * view.GetView();
    SCI::Mat4 imvp = mvp.Inverse();

    std::vector<int>         * conn = &(data->conn);
    std::vector<SCI::Vex3> * vert = &(data->vert);

    SCI::Vex3 bbmid = (bbmin+bbmax)/2.0f;
    SCI::Vex3 bbsiz = (bbmax-bbmin);
    float scl = 2.0f / max( bbsiz.x, max( bbsiz.y, bbsiz.z ) );

    glScalef( scl, scl, scl );
    glTranslatef( -bbmid.x, -bbmid.y, -bbmid.z );

    glPointSize(7.0f);
    glLineWidth(1.0f);

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    for(int pass = 0; pass <= 1; pass++){
        if(pass==0) glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        if(pass==0) glColor3f(0.8f,0.8f,0.8f);
        if(pass==1) glPolygonMode( GL_FRONT_AND_BACK, GL_POINT );
        if(pass==1) glColor3f(0,0,0);
        glBegin(GL_TRIANGLES);
        for(int i = 0; i < (int)conn->size(); i++){
            if(pass==1) glColor4ubv( (GLubyte*)&col_data[ conn->at(i) ] );
            glVertex3fv( vert->at( conn->at(i) ).data );
        }
        glEnd();
    }
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    glDisable( GL_BLEND );

}


void MeshDisplay::ResetView(){
    view.Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    view.Save("view.txt");
}


bool MeshDisplay::MouseClick(int button, int state, int x, int y){
    view.Save("view.txt");

    mouse_active = false;
    if( isPointInFrame(x,y) && state == Down ){
        mouse_active = true;
    }
    return mouse_active;
}


bool MeshDisplay::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if(button == LeftButton)   view.Rotate(-(float)(dx),(float)(dy));
        if(button == MiddleButton) view.Translate((float)(dx),(float)(dy));
        if(button == RightButton)  view.Zoom((float)(dy));
    }
    return mouse_active;
}
