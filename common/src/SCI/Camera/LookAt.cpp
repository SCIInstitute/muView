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


#include <SCI/Camera/LookAt.h>
#include <GL/oglCommon.h>


#include <stdio.h>

#define DEG2RAD(deg) ((deg)/180.0f*3.14159265f)
#define RAD2DEG(rad) ((rad)*180.0f/3.14159265f)

using namespace SCI;

LookAt::LookAt(){
    mat.LoadIdentity();
    imat.LoadIdentity();
    eye.Set(0,0,0);
    center.Set(0,0,-1);
    up.Set(0,1,0);
}

LookAt::LookAt(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ){
    Vex3 _eye(eyeX,eyeY,eyeZ);
    Vex3 _center(centerX,centerY,centerZ);
    Vex3 _up(upX,upY,upZ);

    Set(_eye,_center,_up);
}

LookAt::LookAt(Vex3 _eye, Vex3 _center, Vex3 _up){
    Set(_eye,_center,_up);
}

void LookAt::Set(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ){
    Vex3 _eye(eyeX,eyeY,eyeZ);
    Vex3 _center(centerX,centerY,centerZ);
    Vex3 _up(upX,upY,upZ);

    Set(_eye,_center,_up);
}

void LookAt::Set(Vex3 _eye, Vex3 _center, Vex3 _up){
    eye = _eye;
    center = _center;
    up = _up;

    Vex3 f = (center - eye).UnitVector();

    up.Normalize();

    Vex3 s = cross(f, up);
    Vex3 u = cross(s, f);

    Mat4 mtmp;

    mtmp.Row(0,Vex4( s,0));
    mtmp.Row(1,Vex4( u,0));
    mtmp.Row(2,Vex4(-f,0));
    mtmp.Row(3,Vex4(0,0,0,1));
    
    mat  = mtmp * Mat4(Mat4::MAT4_TRANSLATE,-eye.x,-eye.y,-eye.z);
    imat = mat.Inverse();
}

void LookAt::Get(Vex3& _eye, Vex3& _center, Vex3& _up){
    _eye = eye;
    _center = center;
    _up = up;
}

void LookAt::Get(float& eyeX, float& eyeY, float& eyeZ, float& centerX, float& centerY, float& centerZ, float& upX, float& upY, float& upZ){
    eyeX = eye.x;
    eyeY = eye.y;
    eyeZ = eye.z;

    centerX = center.x;
    centerY = center.y;
    centerZ = center.z;

    upX = up.x;
    upY = up.y;
    upZ = up.z;
}

LookAt& LookAt::operator=(LookAt& other){
    Set(other.GetEye(),other.GetCenter(),other.GetUp());
    return (*this);
}


void LookAt::MoveForward(float amt){
    Vex3 md = (center - eye).UnitVector() * amt;
    Set(eye+md,center+md,up);
}

void LookAt::MoveBackward(float amt){
    Vex3 md = (center - eye).UnitVector() * amt;
    Set(eye-md,center-md,up);
}

void LookAt::MoveLeft(float amt){
    Vex3 md = cross(up,(center - eye)).UnitVector() * amt;
    Set(eye+md,center+md,up);
}

void LookAt::MoveRight(float amt){
    Vex3 md = cross(up,(center - eye)).UnitVector() * amt;
    Set(eye-md,center-md,up);
}

void LookAt::MoveUp(float amt){
    Vex3 md0 = cross(up,(center - eye)).UnitVector();
    Vex3 md1 = cross(md0,(center - eye)).UnitVector() * amt;
    Set(eye+md1,center+md1,up);
}

void LookAt::MoveDown(float amt){
    Vex3 md0 = cross(up,(center - eye)).UnitVector();
    Vex3 md1 = cross(md0,(center - eye)).UnitVector() * amt;
    Set(eye-md1,center-md1,up);
}

void LookAt::RotateLeftRight(float amt){
    Yaw(amt);
}

void LookAt::RotateUpDown(float amt){
    Pitch(amt);
}

void LookAt::Yaw(float amt){
    Vex3 forward = (center - eye).UnitVector();
    Vex3 right   = cross(forward,up).UnitVector();
        Vex3 nfrwd = forward * ::cosf(DEG2RAD(amt)) + right * ::sinf(DEG2RAD(amt));
    Set(eye,eye+nfrwd,up);
}

void LookAt::Pitch(float amt){
    Vex3 forward = (center - eye).UnitVector();
    Vex3 right   = cross(forward,up).UnitVector();
        Vex3 nfrwd = forward * ::cosf(DEG2RAD(amt)) + up * ::sinf(DEG2RAD(amt));
    Vex3 nup  = cross(right,nfrwd).UnitVector();
    Set(eye,eye+nfrwd,nup);
}

void LookAt::Roll(float amt){
    Vex3 forward = (center - eye).UnitVector();
    Vex3 right   = cross(forward,up).UnitVector();
        Vex3 nup = up * ::cosf(DEG2RAD(amt)) + right * ::sinf(DEG2RAD(amt));
    Set(eye,center,nup);
}


