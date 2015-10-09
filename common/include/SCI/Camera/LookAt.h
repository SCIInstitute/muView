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

#ifndef SCI_CAMERA_LOOKAT_H
#define SCI_CAMERA_LOOKAT_H

#include <SCI/Mat4.h>

namespace SCI {
    class LookAt{
        Mat4 mat;
        Mat4 imat;
        Vex3 eye,center,up;

    public:
        LookAt();
        LookAt(Vex3 eye, Vex3 center, Vex3 up);
        LookAt(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ);

        void Set(Vex3 eye, Vex3 center, Vex3 up);
        void Set(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ);

        void Get(Vex3& eye, Vex3& center, Vex3& up);
        void Get(float& eyeX, float& eyeY, float& eyeZ, float& centerX, float& centerY, float& centerZ, float& upX, float& upY, float& upZ);

        inline Vex3 Apply(Vex3& p){ return mat * p; }
        inline Vex3 Apply(float x, float y, float z){ return mat * Vex3(x,y,z); }

        inline Vex3 Unapply(Vex3& p){ return imat * p; }
        inline Vex3 Unapply(float x, float y, float z){ return imat * Vex3(x,y,z); }

        inline Vex3 GetEye(){ return eye; }
        inline Vex3 GetCenter(){ return center; }
        inline Vex3 GetUp(){ return up; }
        inline Vex3 GetVD(){ return (center-eye).UnitVector(); }
        inline Vex3 GetRight(){ return cross(center-eye,up).UnitVector(); }

        inline Mat4 GetMatrix(){ return mat; }
        inline Mat4 GetInverseMatrix(){ return imat; }

        LookAt& operator=(LookAt& other);

        void MoveForward(float amt);
        void MoveBackward(float amt);
        void MoveLeft(float amt);
        void MoveRight(float amt);
        void MoveUp(float amt);
        void MoveDown(float amt);
        void RotateLeftRight(float amt);
        void RotateUpDown(float amt);

        void Yaw(float amt);
        void Pitch(float amt);
        void Roll(float amt);
    };
}

#endif // SCI_CAMERA_LOOKAT_H
