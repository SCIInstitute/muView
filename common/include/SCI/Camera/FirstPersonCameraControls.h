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

#ifndef SCI_CAMERA_FIRSTPERSONCAMERACONTROLS_H
#define SCI_CAMERA_FIRSTPERSONCAMERACONTROLS_H

#include <SCI/Camera/CameraControls.h>
#include <SCI/Camera/LookAt.h>

namespace SCI {

    class FirstPersonCameraControls : public CameraControls {

    protected:
        float angleX,angleY;
        LookAt lat;

    public:
        FirstPersonCameraControls(Vex3 startPos = Vex3());
        ~FirstPersonCameraControls(void);

        Mat4 GetView(){ return lat.GetMatrix(); }

        void ResetView(Vex3 startPos = Vex3());


        virtual void LookLeft(float amount);
        virtual void LookRight(float amount);
        virtual void LookUp(float amount);
        virtual void LookDown(float amount);

        virtual void MoveForward(float amount);
        virtual void MoveBackward(float amount);
        virtual void MoveLeft(float amount);
        virtual void MoveRight(float amount);

        virtual void MoveUp(float amount);
        virtual void MoveDown(float amount);

        void SetEye(Vex3 pos);

        Vex3 GetEye(){ return lat.GetEye(); }
        Vex3 GetVD(){ return lat.GetVD(); }
        Vex3 GetUp(){ return lat.GetUp(); }

        Vex3 GetForward(){ return Vex3(lat.GetVD().x, 0, lat.GetVD().z); }
        Vex3 GetRight()     {     return cross(GetForward(),Vex3(0,1,0)); }

        bool Save(const char * fname);
        bool Load(const char * fname);
    };
}

#endif // SCI_CAMERA_FIRSTPERSONCAMERACONTROLS_H
