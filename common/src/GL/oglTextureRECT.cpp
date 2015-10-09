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

#include <GL/oglTextureRECT.h>
#include <iostream>

using namespace oglWidgets;

#ifndef __ANDROID__

oglTextureRECT::oglTextureRECT(void){
    extCheck = false;
}

oglTextureRECT::~oglTextureRECT(void){ }

bool oglTextureRECT::_CheckExtensions(){
    if(extCheck){ return true; }
    if(glExtensionSupported("GL_ARB_texture_rectangle")){
        target = GL_TEXTURE_RECTANGLE_ARB;
        return true;
    }
#ifndef __APPLE__
    if(glExtensionSupported("GL_NV_texture_rectangle")){
        target = GL_TEXTURE_RECTANGLE_NV;
        return true;
    }
#endif
    std::cerr << "oglTextureRECT:" << "Rectangular textures not supported by graphics card" << std::endl << std::flush;
    return false;
}

bool oglTextureRECT::TexImage2D(GLenum iformat, GLsizei w, GLsizei h, GLenum format, GLenum type, const void* pixels){
    if(!_CheckExtensions()){ return false; }

    glGetError(); // clear errors

    Enable();
    Bind();
        glTexImage2D(target,0,iformat,w,h,0,format,type,pixels);
        glTexParameteri(target,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(target,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(target,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(target,GL_TEXTURE_WRAP_T,wrapt);
    Disable();

    GLenum err = glGetError();

    if(err != 0){
        std::cerr << "oglTextureRECT:" << glErrorString(err) << std::endl << std::flush;
        return false;
    }
    return true;
}

bool oglTextureRECT::TexSubImage2D(GLint xoffset, GLint yoffset, GLsizei w, GLsizei h, GLenum format, GLenum type, const void *pixels){
    if(!_CheckExtensions()){ return false; }

    glGetError(); // clear errors

    Enable();
    Bind();
        glTexSubImage2D(target,0,xoffset,yoffset,w,h,format,type,pixels);
        glTexParameteri(target,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(target,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(target,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(target,GL_TEXTURE_WRAP_T,wrapt);
    Disable();

    GLenum err = glGetError();

    if(err != 0){
        std::cerr << "oglTextureRECT:" << glErrorString(err) << std::endl << std::flush;
        return false;
    }
    return true;
}


bool oglTextureRECT::CopyTexImage2D(GLenum iformat, GLint x, GLint y, GLsizei w, GLsizei h){
    if(!_CheckExtensions()){ return false; }

    glGetError(); // clear errors

    Enable();
    Bind();
        glCopyTexImage2D(target,0,iformat,x,y,w,h,0);
        glTexParameteri(target,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(target,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(target,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(target,GL_TEXTURE_WRAP_T,wrapt);
    Disable();

    GLenum err = glGetError();

    if(err != 0){
        std::cerr << "oglTextureRECT:" << glErrorString(err) << std::endl << std::flush;
        return false;
    }
    return true;
}

bool oglTextureRECT::CopyTexSubImage2D(GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei w, GLsizei h){
    if(!_CheckExtensions()){ return false; }

    glGetError(); // clear errors

    Enable();
    Bind();
        glCopyTexSubImage2D(target,0,xoffset,yoffset,x,y,w,h);
        glTexParameteri(target,GL_TEXTURE_MIN_FILTER,minfilter);
        glTexParameteri(target,GL_TEXTURE_MAG_FILTER,magfilter);
        glTexParameteri(target,GL_TEXTURE_WRAP_S,wraps);
        glTexParameteri(target,GL_TEXTURE_WRAP_T,wrapt);
    Disable();

    GLenum err = glGetError();

    if(err != 0){
        std::cerr << "oglTextureRECT:" << glErrorString(err) << std::endl << std::flush;
        return false;
    }
    return true;
}

#endif
