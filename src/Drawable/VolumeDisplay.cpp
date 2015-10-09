//#include <GL/glew.h>

#include <Drawable/VolumeDisplay.h>
#include <GL/oglCommon.h>


using namespace Drawable;


VolumeDisplay::VolumeShader::VolumeShader(){ }

void VolumeDisplay::VolumeShader::Load( ){
    oglWidgets::oglShader::LoadFromFile( "vol.vert.glsl", "vol.frag.glsl" );
}

VolumeDisplay::VolumeDisplay(){
    proj.Set( 40.0f,40.0f, 0.1f,100.0f);
    view.Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    view.Load("view.txt");
    data         = 0;
    vol_data     = 0;
    curr_node    = 0;
    updated      = false;
    mouse_active = false;
}

void VolumeDisplay::Initialize( Data::ProxyData & _data, Drawable::Provenance & _prov ){
    data     = &_data;
    prov     = &_prov;
    vol_data = new unsigned int[ data->GetX()*data->GetY()*data->GetZ() ];
    shader.Load();
}

void VolumeDisplay::Vertex( SCI::Mat4 & imvp, SCI::Vex3 pnt, SCI::Vex3 volmin, SCI::Vex3 volmax ){
    SCI::Vex3 p = imvp * pnt;
    SCI::Vex3 t = (p-volmin)/(volmax-volmin);
    glTexCoord3fv(t.data);
    glVertex3fv(p.data);
}

unsigned int * VolumeDisplay::ClearVolume( ){
    return (unsigned int*)memset( vol_data, 0, sizeof(int)*data->GetX()*data->GetY()*data->GetZ() );
}

void VolumeDisplay::UpdateVolume( ){
    vol_tex.SetMinMagFilter( GL_LINEAR, GL_LINEAR );
    vol_tex.SetWrapR( GL_CLAMP );
    vol_tex.SetWrapS( GL_CLAMP );
    vol_tex.SetWrapT( GL_CLAMP );
    vol_tex.TexImage3D( GL_RGBA, data->GetX(), data->GetY(), data->GetZ(), GL_RGBA, GL_UNSIGNED_BYTE, vol_data );
}

void VolumeDisplay::Draw(){
    if(data == 0) return;

    if( (prov->GetCurrentNode() != curr_node) || (!updated && curr_node->isDone()) ){
        curr_node = prov->GetCurrentNode();
        updated   = curr_node->isDone();
        curr_node->ColorData( ClearVolume( ) );
        UpdateVolume();
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

    SCI::Vex3 minp = SCI::VEX3_MAX;
    SCI::Vex3 maxp = SCI::VEX3_MIN;
    for(int i = 0; i < 8; i++){
        SCI::Vex3 outp = mvp * SCI::Vex3( minmax[i%2].x, minmax[(i/2)%2].y, minmax[(i/4)%2].z );;
        minp = SCI::Min( minp, outp );
        maxp = SCI::Max( maxp, outp );
    }
    minp = SCI::Max( minp, SCI::Vex3(-1,-1,-1) );
    maxp = SCI::Min( maxp, SCI::Vex3( 1, 1, 1) );

    glDisable(GL_DEPTH_TEST);

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    int slices = 128;
    shader.Enable();
        vol_tex.Enable();
        vol_tex.Bind();
            glBegin(GL_QUADS);
            for(int i = slices-1; i>=0; i--){
                float curz = SCI::lerp( minp.z, maxp.z, (float)i/(float)(slices-1) );
                Vertex( imvp, SCI::Vex3( minp.x, minp.y, curz ), minmax[0], minmax[1] );
                Vertex( imvp, SCI::Vex3( maxp.x, minp.y, curz ), minmax[0], minmax[1] );
                Vertex( imvp, SCI::Vex3( maxp.x, maxp.y, curz ), minmax[0], minmax[1] );
                Vertex( imvp, SCI::Vex3( minp.x, maxp.y, curz ), minmax[0], minmax[1] );
            }
            glEnd();
        vol_tex.Disable();
    shader.Disable();

    glDisable( GL_BLEND );
}


void VolumeDisplay::ResetView(){
    view.Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    view.Save("view.txt");
}


bool VolumeDisplay::MouseClick(int button, int state, int x, int y){
    view.Save("view.txt");

    mouse_active = false;
    if( isPointInFrame(x,y) && state == Down ){
        mouse_active = true;
    }
    return mouse_active;
}


bool VolumeDisplay::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if(button == LeftButton)   view.Rotate(-(float)(dx),(float)(dy));
        if(button == MiddleButton) view.Translate((float)(dx),(float)(dy));
        if(button == RightButton)  view.Zoom((float)(dy));
    }
    return mouse_active;
}
