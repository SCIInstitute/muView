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

#include <SCI/Camera/OrthoProjection.h>
#include <GL/oglCommon.h>

using namespace SCI;


OrthoProjection::OrthoProjection(float _left, float _right, float _bottom, float _top, float _znear, float _zfar){
    left = _left;
    right = _right;
    bottom = _bottom;
    top = _top;
    znear = _znear;
    zfar = _zfar;

    //proj.LoadZero();

        SCI::Vex4 col0(2 / (right - left),0,0,0);
        SCI::Vex4 col1(0,2 / (top - bottom),0,0);
        SCI::Vex4 col2(0,0,-2 / (zfar - znear),0);
        SCI::Vex4 col3;

    col3.x = - (right + left) / (right - left);
    col3.y = - (top + bottom) / (top - bottom);
    col3.z = - (zfar + znear) / (zfar - znear);
    col3.w = 1.0f;

    proj.Column(0,col0);
    proj.Column(1,col1);
    proj.Column(2,col2);
    proj.Column(3,col3);

    //proj.data[0] = 2 / (right - left);
    //proj.data[5] = 2 / (top - bottom);
    //proj.data[10] = -2 / (zfar - znear);
    //proj.data[12] = - (right + left) / (right - left);
    //proj.data[13] = - (top + bottom) / (top - bottom);
    //proj.data[14] = - (zfar + znear) / (zfar - znear);
    //proj.data[15] = 1.0f;

    iproj = proj.Inverse();

    emit( Modified() );
}


OrthoProjection::~OrthoProjection(void){

}

void OrthoProjection::Set(float _left, float _right, float _bottom, float _top, float _znear, float _zfar){
    left = _left;
    right = _right;
    bottom = _bottom;
    top = _top;
    znear = _znear;
    zfar = _zfar;

    //proj.LoadZero();
    //proj.data[0] = 2 / (right - left);
    //proj.data[5] = 2 / (top - bottom);
    //proj.data[10] = -2 / (zfar - znear);
    //proj.data[12] = - (right + left) / (right - left);
    //proj.data[13] = - (top + bottom) / (top - bottom);
    //proj.data[14] = - (zfar + znear) / (zfar - znear);
    //proj.data[15] = 1.0f;

        SCI::Vex4 col0(2 / (right - left),0,0,0);
        SCI::Vex4 col1(0,2 / (top - bottom),0,0);
        SCI::Vex4 col2(0,0,-2 / (zfar - znear),0);
        SCI::Vex4 col3;

    col3.x = - (right + left) / (right - left);
    col3.y = - (top + bottom) / (top - bottom);
    col3.z = - (zfar + znear) / (zfar - znear);
    col3.w = 1.0f;

    proj.Column(0,col0);
    proj.Column(1,col1);
    proj.Column(2,col2);
    proj.Column(3,col3);

    iproj = proj.Inverse();

    emit( Modified() );
}


void OrthoProjection::glSetProjection(bool loadIdent){

    glMatrixMode(GL_PROJECTION);
    if(loadIdent){ glLoadIdentity(); }
    glMultMatrixf( proj.data );

}
