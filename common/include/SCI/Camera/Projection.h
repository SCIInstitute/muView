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

#ifndef SCI_CAMERA_PROJECTION_H
#define SCI_CAMERA_PROJECTION_H

#include <SCI/Mat4.h>

#include <QObject>

namespace SCI {

    class Projection : public QObject {

        Q_OBJECT

    signals:
        void Modified();

    public:
        Projection(void){ }
        virtual ~Projection(void){ }

        virtual void glSetProjection(bool loadIdent) = 0;

        virtual SCI::Mat4 GetMatrix() = 0;
        virtual SCI::Mat4 GetInverseMatrix() = 0;
    };
}

#endif // SCI_CAMERA_PROJECTION_H
