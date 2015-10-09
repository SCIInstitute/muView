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

#ifndef SCI_CAMERA_FRUSTUMPROJECTION_H
#define SCI_CAMERA_FRUSTUMPROJECTION_H

#include <SCI/Camera/Projection.h>

namespace SCI {

    class FrustumProjection : public Projection {

        Q_OBJECT

    public slots:
        void SetHFOV(double fov){ Set(hfov,fov,hither,yon); }
        void SetVFOV(double fov){ Set(fov,vfov,hither,yon); }
        void SetHither(double znear){ Set(hfov,vfov,znear,yon); }
        void SetYon(double zfar){ Set(hfov,vfov,hither,zfar); }

    protected:
        SCI::Mat4 proj, iproj;

        float hfov,vfov,hither,yon;

        void _Set(float left, float right, float bottom, float top, float znear, float zfar);
    public:
        FrustumProjection(float hfov = 45.0f, float vfov = 45.0f, float znear = 1.0f, float zfar = 100.0f);
        FrustumProjection(float hfov, int w, int h, float znear, float zfar);
        ~FrustumProjection(void);

        void Set(float hfov, int w, int h, float znear, float zfar);
        void Set(float hfov, float vfov, float znear, float zfar);

        //void SetHither(float znear){ Set(hfov,vfov,znear,yon); }
        //void SetYon(float zfar){ Set(hfov,vfov,hither,zfar); }

        virtual void glSetProjection(bool loadIdent = true);

        bool inFrustum(SCI::Vex3 p3d);

        virtual SCI::Mat4 GetMatrix()        {    return proj;    }
        virtual SCI::Mat4 GetInverseMatrix()    {    return iproj;    }

        float GetHFOV(){    return hfov;    }
        float GetVFOV(){    return vfov;    }
        float GetHither(){    return hither;    }
        float GetYon(){        return yon;        }

        FrustumProjection& operator=(FrustumProjection& other){
            Set(other.hfov,other.vfov,other.hither,other.yon);
            return (*this);
        }

        void glDrawFrustumVisualization();

        bool Save(const char * fname);
        bool Load(const char * fname);
    };
}

#endif // SCI_CAMERA_FRUSTUMPROJECTION_H
