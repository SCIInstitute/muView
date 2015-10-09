#ifndef GL_GLSLSHADER_H
#define GL_GLSLSHADER_H

#include <GL/oglCommon.h>

namespace oglWidgets {
    class oglShader {

    public:
        oglShader();
        ~oglShader();

        virtual bool Unload( );

        bool Enable();
        bool Disable();

        int GetAttribLocation ( const char * attr_name );
        int GetUniformLocation( const char * attr_name );

        bool isLoaded();

    protected:

        virtual bool LoadFromSource( const char * vert_prog, const char * frag_prog );
        virtual bool LoadFromFile( const char * vert_file, const char * frag_file );

        static unsigned int LoadShader(unsigned int shaderType, const char* pSource);
        static unsigned int CreateProgram(unsigned int vertexShader, unsigned int fragmentShader);

        unsigned int vertexShader;
        unsigned int pixelShader;
        unsigned int program;
        bool loaded;
    };



}

#endif // GL_GLSLSHADER_H
