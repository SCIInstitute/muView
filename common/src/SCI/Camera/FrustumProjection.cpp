/*
**  Common Library
**  Copyright (C) 2013  Paul Rosen
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <SCI/Camera/FrustumProjection.h>

#include <GL/oglCommon.h>

#include <math.h>

#define DEG2RAD(deg) ((deg)/180.0f*3.14159265f)
#define RAD2DEG(rad) ((rad)*180.0f/3.14159265f)
using namespace SCI;


FrustumProjection::FrustumProjection(float _hfov, float _vfov, float znear, float zfar){
    Set(_hfov,_vfov,znear,zfar);
}

FrustumProjection::FrustumProjection(float _hfov, int _w, int _h, float znear, float zfar){
    Set(_hfov,_w,_h,znear,zfar);
}

FrustumProjection::~FrustumProjection(void){

}

void FrustumProjection::_Set(float left, float right, float bottom, float top, float znear, float zfar){

    SCI::Vex4 col0(2.0f * znear / (right - left),0,0,0);
    SCI::Vex4 col1(0,2.0f * znear / (top - bottom),0,0);
    SCI::Vex4 col2;
    SCI::Vex4 col3(0,0,-2.0f * zfar * znear / (zfar - znear),0);

    col2.x =  (right + left) / (right - left);
    col2.y =  (top + bottom) / (top - bottom);
    col2.z = -(zfar + znear) / (zfar - znear);
    col2.w = -1.0f;

    proj.Column(0,col0);
    proj.Column(1,col1);
    proj.Column(2,col2);
    proj.Column(3,col3);

    iproj = proj.Inverse();

    emit( Modified() );
}

void FrustumProjection::Set(float _hfov, int w, int h, float znear, float zfar){

    float f = ((float)w / 2.0f) / tanf(DEG2RAD(_hfov/2.0f));
    float _vfov = 2.0f*RAD2DEG(atanf(((float)h / 2.0f) / f));

    Set(_hfov,_vfov,znear,zfar);
}

void FrustumProjection::Set(float _hfov, float _vfov, float znear, float zfar){
    hfov = _hfov;
    vfov = _vfov;
    hither = znear;
    yon = zfar;

    float thfov = tanf(DEG2RAD(hfov/2.0f))*znear;
    float tvfov = tanf(DEG2RAD(vfov/2.0f))*znear;

    _Set(-thfov,thfov,-tvfov,tvfov,znear,zfar);

}


void FrustumProjection::glSetProjection(bool loadIdent){
    glMatrixMode(GL_PROJECTION);
    if(loadIdent){ glLoadIdentity(); }
    glMultMatrixf(proj.data);
}

bool FrustumProjection::inFrustum(SCI::Vex3 p3d){
        SCI::Vex3 p_clp = proj * p3d;

    bool b0 = p_clp.x >= -1.0f && p_clp.x <= 1.0f;
    bool b1 = p_clp.y >= -1.0f && p_clp.y <= 1.0f;
    bool b2 = p_clp.z >= -1.0f && p_clp.z <= 1.0f;

    return (b0 && b1 && b2);
}

void FrustumProjection::glDrawFrustumVisualization(){

        SCI::Vex3 pn0 = iproj * SCI::Vex3(-1,-1,-1);
        SCI::Vex3 pn1 = iproj * SCI::Vex3( 1,-1,-1);
        SCI::Vex3 pn2 = iproj * SCI::Vex3( 1, 1,-1);
        SCI::Vex3 pn3 = iproj * SCI::Vex3(-1, 1,-1);

        SCI::Vex3 pf0 = iproj * SCI::Vex3(-1,-1, 1);
        SCI::Vex3 pf1 = iproj * SCI::Vex3( 1,-1, 1);
        SCI::Vex3 pf2 = iproj * SCI::Vex3( 1, 1, 1);
        SCI::Vex3 pf3 = iproj * SCI::Vex3(-1, 1, 1);

    glBegin(GL_LINES);
        glVertex3fv(pn0.data);    glVertex3fv(pn1.data);
        glVertex3fv(pn1.data);    glVertex3fv(pn2.data);
        glVertex3fv(pn2.data);    glVertex3fv(pn3.data);
        glVertex3fv(pn3.data);    glVertex3fv(pn0.data);

        glVertex3fv(pf0.data);    glVertex3fv(pf1.data);
        glVertex3fv(pf1.data);    glVertex3fv(pf2.data);
        glVertex3fv(pf2.data);    glVertex3fv(pf3.data);
        glVertex3fv(pf3.data);    glVertex3fv(pf0.data);

        glVertex3fv(pn0.data);    glVertex3fv(pf0.data);
        glVertex3fv(pn1.data);    glVertex3fv(pf1.data);
        glVertex3fv(pn2.data);    glVertex3fv(pf2.data);
        glVertex3fv(pn3.data);    glVertex3fv(pf3.data);
    glEnd();

    glBegin(GL_POINTS);
    glVertex3f(0,0,0);
    glEnd();
}


bool FrustumProjection::Save(const char * fname){
    FILE * outfile = fopen(fname,"w");
    if(!outfile){ return false;}
    fprintf( outfile, "hfov:    %f\n", hfov  );
    fprintf( outfile, "vfov:    %f\n", vfov  );
    fprintf( outfile, "hither:    %f\n", hither);
    fprintf( outfile, "yon:        %f\n", yon     );
    fclose(outfile);
    return true;
}

bool FrustumProjection::Load(const char * fname){
    FILE * infile = fopen(fname,"r");
    if(!infile){ return false;}
    if( fscanf(infile, " %*s %f ", &hfov  ) != 1 ) printf("WARNING: FrustumProjection::Load Error\n");
    if( fscanf(infile, " %*s %f ", &vfov  ) != 1 ) printf("WARNING: FrustumProjection::Load Error\n");
    if( fscanf(infile, " %*s %f ", &hither) != 1 ) printf("WARNING: FrustumProjection::Load Error\n");
    if( fscanf(infile, " %*s %f ", &yon    )   != 1 ) printf("WARNING: FrustumProjection::Load Error\n");
    Set( hfov, vfov, hither, yon );
    fclose(infile);
    return true;
}
