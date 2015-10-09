/*
**  Common OpenGL Support Library
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

#ifndef OGLWIDGETS_TEXTURERECT_H
#define OGLWIDGETS_TEXTURERECT_H

#include <GL/oglCommon.h>
#include <GL/oglTexture2D.h>

#ifndef __ANDROID__

namespace oglWidgets {

    class oglTextureRECT : public oglTexture2D
    {
    protected:
        bool extCheck;
        bool _CheckExtensions();

    public:
        oglTextureRECT(void);
        ~oglTextureRECT(void);

        virtual bool TexImage2D(GLenum iformat, GLsizei w, GLsizei h, GLenum format, GLenum type, const void* pixels);
        virtual bool TexSubImage2D(GLint xoffset, GLint yoffset, GLsizei w, GLsizei h, GLenum format, GLenum type, const void *pixels);

        virtual bool CopyTexImage2D(GLenum iformat, GLint x, GLint y, GLsizei w, GLsizei h);
        virtual bool CopyTexSubImage2D(GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei w, GLsizei h);
    };

}

#endif

#endif // OGLWIDGETS_TEXTURERECT_H
