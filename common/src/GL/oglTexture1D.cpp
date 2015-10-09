#include <GL/oglTexture1D.h>
#include <iostream>

using namespace oglWidgets;

#ifndef __ANDROID__

oglTexture1D::oglTexture1D(void) : oglTexture(GL_TEXTURE_1D) {
    wraps = GL_CLAMP;
}

oglTexture1D::~oglTexture1D(void){ }

bool oglTexture1D::SetWrapS(GLint v){
    if(v != GL_NEAREST && v != GL_LINEAR){
        std::cerr << "oglTexture1D:" << "Invalid Value" << std::endl << std::flush;
        return false;
    }
    wraps = v;
    return true;
}

void oglTexture1D::TexImage1D(GLint iformat, GLsizei w, GLenum format, GLenum type, const void *pixels){
    Enable();
    Bind();
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,wraps);
        glTexImage1D(GL_TEXTURE_1D,0,iformat,w,0,format,type,pixels);
    Disable();
}

void oglTexture1D::TexSubImage1D(GLint xoffset, GLsizei w, GLenum format, GLenum type, const void *pixels){
    Enable();
    Bind();
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,wraps);
        glTexSubImage1D(GL_TEXTURE_1D,0,xoffset,w,format,type,pixels);
    Disable();
}

void oglTexture1D::CopyTexImage1D(GLenum iformat, GLint x, GLint y, GLsizei w){
    Enable();
    Bind();
        glCopyTexImage1D(GL_TEXTURE_1D,0,iformat,x,y,w,0);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,wraps);
    Disable();
}

void oglTexture1D::CopyTexSubImage1D(GLint xoffset, GLint x, GLint y, GLsizei w){
    Enable();
    Bind();
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,wraps);
        glCopyTexSubImage1D(GL_TEXTURE_1D,0,xoffset,x,y,w);
    Disable();
}

#endif
