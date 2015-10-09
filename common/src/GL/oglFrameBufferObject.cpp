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

#include <iostream>

#ifdef __ANDROID__
    #define GL_DEL_FB           glDeleteFramebuffers
    #define GL_GEN_FB           glGenFramebuffers
    #define GL_BND_FB           glBindFramebuffer
    #define GL_CHK_FB_STATUS    glCheckFramebufferStatus
    #define GL_TEX_FB           glFramebufferTexture2D
    #define GL_BND_RB           glBindRenderbuffer

    #define GL_TEX_CLAMP                            GL_CLAMP_TO_EDGE

    #define GL_DPT_ATTACHMENT                       GL_DEPTH_ATTACHMENT
    #define GL_COL_ATTACHMENT                       GL_COLOR_ATTACHMENT0
    #define GL_STN_ATTACHMENT                       GL_STENCIL_ATTACHMENT

    #define GL_RB_TARGET                            GL_RENDERBUFFER

    #define GL_FB_TARGET                            GL_FRAMEBUFFER
    #define GL_FB_COMPLETE                          GL_FRAMEBUFFER_COMPLETE
    #define GL_FB_INCOMPLETE_ATTACHMENT             GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT
    #define GL_FB_INCOMPLETE_MISSING_ATTACHMENT     GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT
    #define GL_FB_INCOMPLETE_DIMENSIONS             GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS
    #define GL_FB_INCOMPLETE_FORMATS                0X0001
    #define GL_FB_INCOMPLETE_DRAW_BUFFER            0X0002
    #define GL_FB_INCOMPLETE_READ_BUFFER            0X0003
    #define GL_FB_UNSUPPORTED                       GL_FRAMEBUFFER_UNSUPPORTED
#else
    #include <GL/glew.h>
    #define GL_DEL_FB           glDeleteFramebuffersEXT
    #define GL_GEN_FB           glGenFramebuffersEXT
    #define GL_BND_FB           glBindFramebufferEXT
    #define GL_CHK_FB_STATUS    glCheckFramebufferStatusEXT
    #define GL_TEX_FB           glFramebufferTexture2DEXT
    #define GL_BND_RB           glBindRenderbufferEXT

    #define GL_TEX_CLAMP                            GL_CLAMP

    #define GL_DPT_ATTACHMENT                       GL_DEPTH_ATTACHMENT_EXT
    #define GL_COL_ATTACHMENT                       GL_COLOR_ATTACHMENT0_EXT
    #define GL_STN_ATTACHMENT                       GL_STENCIL_ATTACHMENT_EXT

    #define GL_RB_TARGET                            GL_RENDERBUFFER_EXT

    #define GL_FB_TARGET                            GL_FRAMEBUFFER_EXT
    #define GL_FB_COMPLETE                          GL_FRAMEBUFFER_COMPLETE_EXT
    #define GL_FB_INCOMPLETE_ATTACHMENT             GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT
    #define GL_FB_INCOMPLETE_MISSING_ATTACHMENT     GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT
    #define GL_FB_INCOMPLETE_DIMENSIONS             GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT
    #define GL_FB_INCOMPLETE_FORMATS                GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT
    #define GL_FB_INCOMPLETE_DRAW_BUFFER            GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT
    #define GL_FB_INCOMPLETE_READ_BUFFER            GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT
    #define GL_FB_UNSUPPORTED                       GL_FRAMEBUFFER_UNSUPPORTED_EXT
#endif

#include <GL/oglFrameBufferObject.h>

using namespace oglWidgets;

oglFrameBufferObject::oglFrameBufferObject(void) {
    fbo = 0xffffffff;
    rbo = 0xffffffff;
    width = 0;
    height = 0;
}

oglFrameBufferObject::~oglFrameBufferObject(void){
    if(fbo != 0xffffffff){
        GL_DEL_FB(1, &fbo);
    }
    if(rbo != 0xffffffff){
        GL_DEL_FB(1, &rbo);
    }
    for(int i = 0; i < (int)tex_list.size(); i++){
        if(tex_list[i]){ delete tex_list[i]; }
    }
}



bool oglFrameBufferObject::Init(int _width, int _height){
    if(fbo != 0xffffffff){
        std::cerr << "oglFrameBufferObject: " << "Mulitple Initializations" << std::endl << std::flush;
        return false;
    }

    if(!glExtensionSupported("GL_EXT_framebuffer_object")){
        std::cerr << "oglFrameBufferObject: " << "Frame Buffer Objects (FBOs) Not Supported By Graphics Card" << std::endl << std::flush;
        return false;
    }

    if(!glExtensionSupported("GL_EXT_framebuffer_blit")){
        std::cerr << "oglFrameBufferObject: " << "Frame Buffer BLIT Not Supported By Graphics Card" << std::endl << std::flush;
        return false;
    }

    width  = _width;
    height = _height;

    GL_GEN_FB(1, &fbo);

    return true;
}


bool oglFrameBufferObject::CheckErrors(){
    GL_BND_FB(GL_FB_TARGET, fbo);

    GLenum status = GL_CHK_FB_STATUS(GL_FB_TARGET);
    switch(status){
        case GL_FB_INCOMPLETE_ATTACHMENT:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: INCOMPLETE ATTACHMENT" << std::endl << std::flush;
             break;
        case GL_FB_INCOMPLETE_MISSING_ATTACHMENT:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: INCOMPLETE MISSING ATTACHMENT" << std::endl << std::flush;
             break;
        case GL_FB_INCOMPLETE_DIMENSIONS:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: INCOMPLETE DIMENSIONS" << std::endl << std::flush;
             break;
        case GL_FB_INCOMPLETE_FORMATS:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: INCOMPLETE FORMAT" << std::endl << std::flush;
             break;
        case GL_FB_INCOMPLETE_DRAW_BUFFER:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: INCOMPLETE DRAW BUFFER" << std::endl << std::flush;
             break;
        case GL_FB_INCOMPLETE_READ_BUFFER:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: INCOMPLETE READ BUFFER" << std::endl << std::flush;
             break;
        case GL_FB_UNSUPPORTED:
             std::cerr << "oglFrameBufferObject (CheckErrors): " << "Error: UNSUPPORTED" << std::endl << std::flush;
             break;
    }

    GL_BND_FB(GL_FB_TARGET, 0);

    return (status == GL_FB_COMPLETE);
}

bool oglFrameBufferObject::InsertDepthAttachment(oglTexture2D * tex){

    if(fbo == 0xffffffff){
        std::cerr << "oglFrameBufferObject (InsertDepthAttachment): " << "Not Yet Initialized" << std::endl << std::flush;
        return false;
    }

    if(tex == 0){
        tex = new oglTexture2D();
        tex->SetWrapS(GL_TEX_CLAMP);
        tex->SetWrapT(GL_TEX_CLAMP);
        tex->SetMinFilter(GL_LINEAR);
        tex->SetMagFilter(GL_LINEAR);
        tex->TexImage2D(GL_DEPTH_COMPONENT16, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        tex_list.push_back( tex );
    }

    depthTexture = tex;

    GL_BND_FB(GL_FRAMEBUFFER, fbo);
    GL_TEX_FB(GL_FRAMEBUFFER, GL_DPT_ATTACHMENT, GL_TEXTURE_2D, tex->GetTexID(), 0);
    #ifndef __ANDROID__
        if(colorAttachment.size() == 0){
            glDrawBuffers(1,&(colorAttachment[0]) );
        }
    #endif
    GL_BND_FB(GL_FRAMEBUFFER, 0);

    return true;
}



bool oglFrameBufferObject::InsertColorAttachment(int id, oglTexture2D * tex){

    if(fbo == 0xffffffff){
        std::cerr << "oglFrameBufferObject (InsertColorAttachment): " << "Not Yet Initialized" << std::endl << std::flush;
        return false;
    }

    GLint maxbuffers;
    #ifdef __ANDROID__
        maxbuffers = 1;
    #else
        glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxbuffers);
    #endif
    if((int)colorAttachment.size() >= maxbuffers){
        std::cerr << "oglFrameBufferObject (InsertColorAttachment): " << "Maximum Color Attachments Exceeded" << std::endl << std::flush;
        return false;
    }

    if(tex == 0){
        tex = new oglTexture2D();
        tex->SetWrapS(GL_TEX_CLAMP);
        tex->SetWrapT(GL_TEX_CLAMP);
        tex->SetMinFilter(GL_LINEAR);
        tex->SetMagFilter(GL_LINEAR);
        #ifdef __ANDROID__
            tex->TexImage2D(GL_RGBA, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        #else
            tex->TexImage2D(GL_RGBA8, width, height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL);
        #endif
        tex_list.push_back( tex );
    }

    int color_attachment = GL_COL_ATTACHMENT + id;

    colorTexture.push_back( tex );
    colorAttachment.push_back( color_attachment );

    GL_BND_FB(GL_FRAMEBUFFER, fbo);
    GL_TEX_FB(GL_FRAMEBUFFER, color_attachment, GL_TEXTURE_2D, tex->GetTexID(), 0);
    #ifndef __ANDROID__
        glDrawBuffers((int)colorAttachment.size(),&(colorAttachment[0]));
    #endif
    GL_BND_FB(GL_FRAMEBUFFER, 0);

    return CheckErrors();
}

void oglFrameBufferObject::Enable( bool set_viewport ) const {
    if(fbo == 0xffffffff){
        std::cerr << "oglFrameBufferObject (Enable): " << "Not Yet Initialized" << std::endl << std::flush;
    }
    else{
        GL_BND_FB(GL_FRAMEBUFFER, fbo);
        if(rbo != 0xffffffff){
            GL_BND_RB(GL_RB_TARGET, rbo);
        }
    }


    #ifndef __ANDROID__
        glPushAttrib(GL_VIEWPORT_BIT);
    #endif
    if( set_viewport ){
        glViewport(0,0,GetWidth(),GetHeight());
    }
}

void oglFrameBufferObject::Disable() const {
    #ifndef __ANDROID__
        glPopAttrib();
    #endif

    GL_BND_FB(GL_FRAMEBUFFER, 0);
    GL_BND_RB(GL_RB_TARGET, 0);
}

#ifndef __ANDROID__

void oglFrameBufferObject::Copy(oglFrameBufferObject & dst){
    GL_BND_FB(GL_READ_FRAMEBUFFER_EXT, fbo);
    GL_BND_FB(GL_DRAW_FRAMEBUFFER_EXT, dst.fbo);
        glBlitFramebufferEXT(0, 0, width, height, 0, 0, dst.width, dst.height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_LINEAR);
    GL_BND_FB(GL_READ_FRAMEBUFFER_EXT, 0);
    GL_BND_FB(GL_DRAW_FRAMEBUFFER_EXT, 0);
}

void oglFrameBufferObject::CopyRegion(oglFrameBufferObject & dst, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter){
    GL_BND_FB(GL_READ_FRAMEBUFFER_EXT, fbo);
    GL_BND_FB(GL_DRAW_FRAMEBUFFER_EXT, dst.fbo);
        glBlitFramebufferEXT(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
    GL_BND_FB(GL_READ_FRAMEBUFFER_EXT, 0);
    GL_BND_FB(GL_DRAW_FRAMEBUFFER_EXT, 0);
}

#endif
