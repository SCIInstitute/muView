#include "heart/HeartPointDisplay.h"
#include <GL/oglCommon.h>
#include <iostream>

#if 0
using namespace Drawable;

HeartPointDisplay::HeartPointDisplay(){
    proj.Set( 40.0f,40.0f, 0.1f,100.0f);
    view.Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    view.Load("view.txt");
    data         = 0;
    updated      = false;
    mouse_active = false;
    curdim = -1;
}

void HeartPointDisplay::Initialize( Data::PointData &_data ){
    data     = &_data;

    //std::vector<SCI::Vex3> * vert = &((Data::PointData*)data)->GetVertices();
    std::vector<SCI::Vex3> * vert = &(data->GetVertices());

    col_data = new unsigned int[ data->GetX()*data->GetY()*data->GetZ() ];

    bbmin = SCI::VEX3_MAX;
    bbmax = SCI::VEX3_MIN;

    for(int i = 0; i < (int)vert->size(); i++){
        bbmin = SCI::Min( bbmin, vert->at(i) );
        bbmax = SCI::Max( bbmax, vert->at(i) );
    }

    Recalculate();

}

void HeartPointDisplay::Recalculate( int dim ){

    SCI::Vex3 colormap[10];
    int colN = 0;
    std::vector< float > vals;
    float minv =  FLT_MAX;
    float maxv = -FLT_MAX;

    if(dim == -2){

        colormap[colN++] = SCI::Vex3( 242, 240, 247 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 218, 218, 235 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 188, 189, 220 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 158, 154, 200 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 117, 107, 177 ) / 255.0f;
        colormap[colN++] = SCI::Vex3(  84,  39, 143 ) / 255.0f;

        for(int i = 0; i < data->GetVoxelCount(); i++){
            float median = 0;
            for(int d = 0; d < data->GetDim(); d++){
                median += data->GetElement( i, d ) / (float)data->GetDim();
            }
            float sum_dif = 0;
            for(int d = 0; d < data->GetDim(); d++){
                sum_dif += powf( data->GetElement( i, d ) - median, 2.0f );
            }
            vals.push_back( sqrtf(sum_dif) );
            minv = SCI::Min( minv, vals.back() );
            maxv = SCI::Max( maxv, vals.back() );
        }

    }
    else{
        colormap[colN++] = SCI::Vex3( 254, 229, 217 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 252, 187, 161 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 252, 146, 114 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 251, 106,  74 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 222,  45,  38 ) / 255.0f;
        colormap[colN++] = SCI::Vex3( 165,  15,  21 ) / 255.0f;

        minv = data->GetMinimumValue();
        maxv = data->GetMaximumValue();
        for(int i = 0; i < data->GetVoxelCount(); i++){
            float val = 0;
            if(dim >= 0 && dim < data->GetDim() ){
                val = data->GetElement( i, dim );
            }
            else{
                for(int d = 0; d < data->GetDim(); d++){
                    val += data->GetElement( i, d ) / (float)data->GetDim();
                }
            }
            vals.push_back( val );
        }

    }

    colormap[colN]   = colormap[colN-1];
    for(int i = 0; i < data->GetVoxelCount(); i++){
        float t = (float)colN * (vals[i]-minv) / (maxv-minv);
        col_data[i] = SCI::lerp( colormap[(int)t], colormap[(int)t+1], t-floorf(t) ).UIntColor();
    }

}

unsigned int * HeartPointDisplay::ClearColor(){
    return (unsigned int*)memset( col_data, 0, sizeof(int)*data->GetX()*data->GetY()*data->GetZ() );
}

void HeartPointDisplay::Draw(){
    if(data == 0) return;

    proj.Set( proj.GetHFOV(), width, height, proj.GetHither(), proj.GetYon() );

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

    std::vector<SCI::Vex3> * vert = &(data->GetVertices());

    SCI::Vex3 bbmid = (bbmin+bbmax)/2.0f;
    SCI::Vex3 bbsiz = (bbmax-bbmin);
    float scl = 2.0f / SCI::Max( bbsiz.x, SCI::Max( bbsiz.y, bbsiz.z ) );

    glScalef( scl, scl, scl );
    glTranslatef( -bbmid.x, -bbmid.y, -bbmid.z );

    glEnable(GL_DEPTH_TEST);

    //glLineWidth(1.0f);

    //glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    //glEnable( GL_BLEND );

    glPointSize(4.0f);
    glBegin(GL_POINTS);
    for(int i = 0; i < (int)vert->size(); i++){
        glColor4ubv( (GLubyte*)&col_data[ i ] );
        glVertex3fv( vert->at( i ).data );
    }
    glEnd();

    glPointSize(5.0f);
    glBegin(GL_POINTS);
    glColor3f(0,0,0);
    for(int i = 0; i < (int)vert->size(); i++){
        glVertex3fv( vert->at( i ).data );
    }
    glEnd();

/*
    for(int pass = 0; pass <= 1; pass++){
        if(pass==0) glPointSize(1.0f);
        if(pass==0) glColor3f(0.8f,0.8f,0.8f);
        if(pass==1) glPointSize(4.0f);
        if(pass==1) glColor3f(0,0,0);
        glBegin(GL_POINTS);
        for(int i = 0; i < (int)vert->size(); i++){
            if(pass==1) glColor4ubv( (GLubyte*)&col_data[ i ] );
            glVertex3fv( vert->at( i ).data );
        }
        glEnd();
    }
    */
    //glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    //glDisable( GL_BLEND );

}


void HeartPointDisplay::ResetView(){
    view.Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    view.Save("view.txt");
}


bool HeartPointDisplay::MouseClick(int button, int state, int x, int y){
    view.Save("view.txt");

    mouse_active = false;
    if( isPointInFrame(x,y) && state == Down ){
        mouse_active = true;
    }
    return mouse_active;
}


bool HeartPointDisplay::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if(button == LeftButton)   view.Rotate(-(float)(dx),(float)(dy));
        if(button == MiddleButton) view.Translate((float)(dx),(float)(dy));
        if(button == RightButton)  view.Zoom((float)(dy));
    }
    return mouse_active;
}

bool HeartPointDisplay::Keypress( int key, int x, int y ){
    if(key == '+'){
        curdim++;
    }
    if(key == '-'){
        curdim--;
    }
    Recalculate( curdim );
    std::cout << "current dim: " << curdim << std::endl; fflush(stdout);
    return false;
}
#endif
