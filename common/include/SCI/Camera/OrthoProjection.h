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

#ifndef SCI_CAMERA_ORTHOPROJECTION_H
#define SCI_CAMERA_ORTHOPROJECTION_H

#include <SCI/Camera/Projection.h>

namespace SCI {

    class OrthoProjection :    public Projection {

        Q_OBJECT

    protected:
        float left,right;
        float bottom,top;
        float znear,zfar;

        SCI::Mat4 proj;
        SCI::Mat4 iproj;

    public:
        OrthoProjection(float left = -1, float right = 1, float bottom = -1, float top = 1, float znear= -1, float zfar = 1);
        ~OrthoProjection(void);

        void Set(float left, float right, float bottom, float top, float znear, float zfar);

        virtual void glSetProjection(bool loadIdent = true);

        virtual SCI::Mat4 GetMatrix()        {    return proj;    }
        virtual SCI::Mat4 GetInverseMatrix()    {    return iproj;    }
    };
}

#endif // SCI_CAMERA_ORTHOPROJECTION_H
