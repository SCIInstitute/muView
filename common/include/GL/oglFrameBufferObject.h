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

#ifndef OGLWIDGETS_OGLFRAMEBUFFEROBJECT_H
#define OGLWIDGETS_OGLFRAMEBUFFEROBJECT_H

#include <GL/oglCommon.h>
#include <GL/oglTexture2D.h>

#include <vector>

namespace oglWidgets {

    class oglFrameBufferObject {

    protected:
        GLuint width;
        GLuint height;
        GLuint fbo;
        GLuint rbo;

        std::vector<oglTexture2D*> tex_list;

        oglTexture2D *             depthTexture;
        std::vector<oglTexture2D*> colorTexture;
        std::vector<GLenum>        colorAttachment;

    public:
        bool CheckErrors();

    public:
        oglFrameBufferObject(void);
        ~oglFrameBufferObject(void);

        bool Init(int width, int height);
        bool InsertDepthAttachment(oglTexture2D * tex = 0);
        bool InsertColorAttachment(int id, oglTexture2D * tex = 0);

        void Enable( bool set_viewport = false )  const ;
        void Disable() const ;

        void Copy(oglFrameBufferObject & dst);
        void CopyRegion(oglFrameBufferObject & dst, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);

        inline GLuint               GetWidth()                      const { return width; }
        inline GLuint               GetHeight()                     const { return height; }
        inline oglTexture2D *    GetColorTexture(int id)               { return colorTexture[id]; }
        inline const oglTexture2D *    GetColorTexture(int id)        const { return colorTexture[id]; }
        inline unsigned int        GetColorTextureID(int id)    const { return colorTexture[id]->GetTexID(); }
        inline GLenum        GetColorAttachment(int id)    const { return colorAttachment[id]; }
        inline int            GetColorCount()            const { return (int)colorAttachment.size(); }
        inline const oglTexture2D *    GetDepthTexture()        const { return depthTexture; }
        inline unsigned int        GetDepthTextureID()        const { return depthTexture->GetTexID(); }
        inline oglTexture2D *    GetDepthTexture()                     { return depthTexture; }
        inline GLuint        GetFboId()                      const { return fbo; }

    };

}

#endif // OGLWIDGETS_OGLFRAMEBUFFEROBJECT_H
