#pragma once

#include <GL/oglCommon.h>
#include <GL/oglTexture.h>

#ifndef __ANDROID__

namespace oglWidgets {

    class oglTexture3D : public oglTexture {

    protected:
        GLint wrapr;
        GLint wraps;
        GLint wrapt;

        bool extCheck;
        bool _CheckExtensions();

    public:
        oglTexture3D(void);
        ~oglTexture3D(void);

        bool SetWrapR(GLint v);
        bool SetWrapS(GLint v);
        bool SetWrapT(GLint v);

        virtual void TexImage3D(GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
        virtual void TexSubImage3D(GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);

        virtual void CopyTexSubImage3D(GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);

    };

}

#endif
