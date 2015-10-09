#pragma once

#include <GL/oglCommon.h>
#include <GL/oglTexture.h>

#ifndef __ANDROID__

namespace oglWidgets {

    class oglTexture1D : public oglTexture {

    protected:
        GLint wraps;

    public:
        oglTexture1D(void);
        virtual ~oglTexture1D(void);

        bool SetWrapS(GLint v);

        virtual void TexImage1D(GLint internalformat, GLsizei w, GLenum format, GLenum type, const void *pixels);
        virtual void TexSubImage1D(GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels);

        virtual void CopyTexImage1D(GLenum internalFormat, GLint x, GLint y, GLsizei width);
        virtual void CopyTexSubImage1D(GLint xoffset, GLint x, GLint y, GLsizei width);
    };
}

#endif
