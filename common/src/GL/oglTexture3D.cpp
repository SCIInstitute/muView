#ifndef __ANDROID__

#include <GL/glew.h>
#include <GL/oglTexture3D.h>
#include <iostream>

using namespace oglWidgets;


oglTexture3D::oglTexture3D(void) : oglTexture(GL_TEXTURE_3D) {
    minfilter = GL_NEAREST;
    magfilter = GL_NEAREST;
    wrapr = GL_CLAMP;
    wraps = GL_CLAMP;
    wrapt = GL_CLAMP;
    glTexImage3D = 0;
    glTexSubImage3D = 0;
    glCopyTexSubImage3D = 0;
    extCheck = false;
}

oglTexture3D::~oglTexture3D(void){

}

bool oglTexture3D::_CheckExtensions(){
    if(extCheck){ return true; }
    extCheck = true;
    if(!glExtensionSupported("GL_EXT_texture3D")){
        std::cerr << "oglTexture3D:" << "3D textures not supported by graphics card" << std::endl << std::flush;
        return false;
    }
    return true;
}

bool oglTexture3D::SetWrapR(GLint v){
    if(v != GL_CLAMP && v != GL_REPEAT){
        std::cerr << "oglTexture3D:" << "Invalid Value" << std::endl << std::flush;
        return false;
    }
    wrapr = v;
    return true;
}

bool oglTexture3D::SetWrapS(GLint v){
    if(v != GL_CLAMP && v != GL_REPEAT){
        std::cerr << "oglTexture3D:" << "Invalid Value" << std::endl << std::flush;
        return false;
    }
    wraps = v;
    return true;
}

bool oglTexture3D::SetWrapT(GLint v){
    if(v != GL_CLAMP && v != GL_REPEAT){
        std::cerr << "oglTexture3D:" << "Invalid Value" << std::endl << std::flush;
        return false;
    }
    wrapt = v;
    return true;
}

void oglTexture3D::TexImage3D(GLint iformat, GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, const void *pixels){
    Enable();
    Bind();
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_R,wrapr);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T,wrapt);
        glTexImage3D(GL_TEXTURE_3D,0,iformat,w,h,d,0,format,type,pixels);
    Disable();
}

void oglTexture3D::TexSubImage3D(GLint xoffset, GLint yoffset, GLint zoffset, GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, const void *pixels){
    Enable();
    Bind();
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_R,wrapr);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T,wrapt);
        glTexSubImage3D(GL_TEXTURE_3D,0,xoffset,yoffset,zoffset,w,h,d,format,type,pixels);
    Disable();
}

void oglTexture3D::CopyTexSubImage3D(GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei w, GLsizei h){
    Enable();
    Bind();
        glCopyTexSubImage3D(GL_TEXTURE_3D,0,xoffset,yoffset,zoffset,x,y,w,h);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_R,wrapr);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T,wrapt);
    Disable();
}


#endif
