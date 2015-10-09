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

#include <SCI/Camera/ThirdPersonCameraControls.h>

using namespace SCI;

ThirdPersonCameraControls::ThirdPersonCameraControls(float startAngleX, float startAngleY, float startDistance, Vex3 startCenter, Vex3 _up){

    up = _up;

    DistanceToCenter = startDistance;

    AngleX = startAngleX;
    AngleY = startAngleY;

    DirectionToEye = Mat4(Mat4::MAT4_ROTATION,AngleX,0,1,0) * Mat4(Mat4::MAT4_ROTATION,180.0f-AngleY,1,0,0) * (-up);

    Center = startCenter;

    _RecalcView();

}

ThirdPersonCameraControls::~ThirdPersonCameraControls(void){

}

void ThirdPersonCameraControls::Set(float startAngleX, float startAngleY, float startDistance, Vex3 startCenter, Vex3 _up){

    up = _up;

    DistanceToCenter = startDistance;

    AngleX = startAngleX;
    AngleY = startAngleY;

    Center = startCenter;

    _RecalcView();

}

void ThirdPersonCameraControls::_RecalcView(){
    DirectionToEye = Mat4(Mat4::MAT4_ROTATION,AngleX,0,1,0) * Mat4(Mat4::MAT4_ROTATION,180.0f-AngleY,1,0,0) * (-up);

    cs_z = (Vex3(0,0,0) - (DirectionToEye.UnitVector()*DistanceToCenter)).UnitVector();
    cs_x = cross(cs_z,Vex3(0,1,0)).UnitVector();
    cs_y = cross(cs_x,cs_z).UnitVector();

    lat.Set(DirectionToEye.UnitVector()*DistanceToCenter+Center,Center,cs_y);
}

Mat4 ThirdPersonCameraControls::GetView(){
    return lat.GetMatrix();
}

Vex3 ThirdPersonCameraControls::GetEye(){
    return DirectionToEye.UnitVector()*DistanceToCenter+Center;
}

Vex3 ThirdPersonCameraControls::GetVD(){
    return (Vex3(0,0,0) - (DirectionToEye.UnitVector()*DistanceToCenter)).UnitVector();
}


Vex3 ThirdPersonCameraControls::GetRight(){
    return cross(GetVD(),Vex3(0,1,0)).UnitVector();
}

Vex3 ThirdPersonCameraControls::GetCenter(){ return Center; }

Vex3 ThirdPersonCameraControls::GetUp(){
    cs_x = cross(cs_z,Vex3(0,1,0)).UnitVector();
    return cross(cs_x,GetVD()).UnitVector();
}


void ThirdPersonCameraControls::Rotate(float amtX, float amtY){
    AngleX += amtX/4.0f;
    AngleY += amtY/4.0f;
    AngleY = Clamp(AngleY,1.0f,179.0f);
    _RecalcView();
}

void ThirdPersonCameraControls::Translate(float amtX, float amtY){
    Center -= cs_x * amtX*DistanceToCenter/700.0f;
    Center -= cs_y * amtY*DistanceToCenter/700.0f;
    _RecalcView();
}

void ThirdPersonCameraControls::Zoom(float amt){
    DistanceToCenter *= powf(1.01f,amt);
    if(DistanceToCenter < 0.1f){
        Center -= cs_z * (float)amt / 20.0f;
        DistanceToCenter = 0.1f;
    }
    _RecalcView();
}

bool ThirdPersonCameraControls::Save(const char* fname){
    FILE* outfile;
    //if(fopen_s(&outfile,fname,"w") != 0){ return false; }
    if( (outfile=fopen(fname,"w")) == 0){ return false; }
    fprintf(outfile,"DistanceToCenter %f\n",DistanceToCenter);
    fprintf(outfile,"AngleX %f\n",AngleX);
    fprintf(outfile,"AngleY %f\n",AngleY);
    fprintf(outfile,"Center %f %f %f\n",Center.x,Center.y,Center.z);
    fclose(outfile);
    return true;
}

bool ThirdPersonCameraControls::Load(const char* fname){
    FILE* infile;
    //if(fopen_s(&infile,fname,"r") != 0){ return false; }
    if( (infile=fopen(fname,"r")) == 0){ return false; }
    if( fscanf(infile," %*s %f ",      &DistanceToCenter)             != 1 ) printf("WARNING: ThirdPersonCameraControls::Load Error\n");
    if( fscanf(infile," %*s %f ",      &AngleX)                       != 1 ) printf("WARNING: ThirdPersonCameraControls::Load Error\n");
    if( fscanf(infile," %*s %f ",      &AngleY)                       != 1 ) printf("WARNING: ThirdPersonCameraControls::Load Error\n");
    if( fscanf(infile," %*s %f %f %f ",&Center.x,&Center.y,&Center.z) != 3 ) printf("WARNING: ThirdPersonCameraControls::Load Error\n");
    fclose(infile);
    _RecalcView();
    return true;
}

void ThirdPersonCameraControls::InterpolateView(ThirdPersonCameraControls & v0, ThirdPersonCameraControls & v1, float t){
    Vex3  cen  = lerp(v0.Center, v1.Center, t);
    float angX = lerp(v0.AngleX, v1.AngleX, t);
    float angY = lerp(v0.AngleY, v1.AngleY, t);
    float dist = lerp(v0.DistanceToCenter, v1.DistanceToCenter, t);

    Set( angX, angY, dist, cen, Vex3(0,1,0) );

    v0.Save("tmp0.txt");
    v1.Save("tmp1.txt");
    Save("tmp2.txt");
}
