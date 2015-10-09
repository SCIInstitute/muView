#ifndef __ANDROID__
    #include <GL/glew.h>
#endif

#include <GL/oglShader.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#ifndef WIN32
#ifndef __APPLE__
    #include <malloc.h>
#endif
#endif

#ifdef __APPLE__
    #include <stdlib.h>
#endif

using namespace oglWidgets;


#ifdef __ANDROID__
    #define GLSHADERSOURCE  glShaderSource
    #define GLCOMPILESHADER glCompileShader
#else
    #define GLSHADERSOURCE  glShaderSourceARB
    #define GLCOMPILESHADER glCompileShaderARB
#endif


oglShader::oglShader(){
    vertexShader = 0;
    pixelShader = 0;
    program = 0;
    loaded = false;
}

oglShader::~oglShader(){
    Unload();
}

bool oglShader::Unload( ){
    if( program != 0 && vertexShader != 0 ){
        glDetachShader( program, vertexShader );
        glDeleteShader( vertexShader );
        vertexShader = 0;
    }
    if( program != 0 && pixelShader != 0 ){
        glDetachShader( program, pixelShader );
        glDeleteShader( pixelShader );
        pixelShader = 0;
    }
    if ( program != 0 ){
        glDeleteProgram(program);
        program = 0;
    }
    return true;
}

bool oglShader::isLoaded(){
    return loaded;
}

bool oglShader::LoadFromSource( const char * vert_prog, const char * frag_prog ){

    loaded = false;

    checkGlError("Unknown");

    Unload();
    checkGlError("Unload");

    std::cout << "oglShader:" << "Loading Vertex Shader" << std::endl << std::flush;
    std::cout << vert_prog << std::endl << std::flush;
    vertexShader = LoadShader(GL_VERTEX_SHADER, vert_prog);
    if (!vertexShader) {
        return false;
    }

    std::cout << "oglShader:" << "Loading Fragment Shader" << std::endl << std::flush;
    std::cout << "oglShader:" << frag_prog << std::endl << std::flush;
    pixelShader = LoadShader(GL_FRAGMENT_SHADER, frag_prog);
    if (!pixelShader) {
        return false;
    }

    program = CreateProgram( vertexShader, pixelShader );
    if( !program ){
        return false;
    }

    return (loaded = true);
}

bool oglShader::LoadFromFile( const char * vert_file, const char * frag_file ){
    FILE * vert_infile = fopen( vert_file, "r" );
    if( vert_infile == 0 ){
        std::cerr << "oglShader:" << "Cannot read shader " << vert_file << std::endl << std::flush;
        return false;
    }

    FILE * frag_infile = fopen( frag_file, "r" );
    if( frag_infile == 0 ){
        fclose(vert_infile);
        std::cerr << "oglShader:" << "Cannot read shader " << frag_file << std::endl << std::flush;
        return false;
    }

    char lbuf[1024];
    char * vert_buf = (char*)malloc(1024);
    char * frag_buf = (char*)malloc(1024);

    vert_buf[0] = 0;
    frag_buf[0] = 0;

    while(!feof(vert_infile)){
        fgets( lbuf, 1024, vert_infile );
        strcat( vert_buf, lbuf );
        vert_buf = (char*)realloc( vert_buf, strlen(vert_buf) + 1024 );
    }

    while(!feof(frag_infile)){
        fgets( lbuf, 1024, frag_infile );
        strcat( frag_buf, lbuf );
        frag_buf = (char*)realloc( frag_buf, strlen(frag_buf) + 1024 );
    }

    bool ret = LoadFromSource( vert_buf, frag_buf );

    free( vert_buf );
    free( frag_buf );

    return ret;
}

GLuint oglShader::CreateProgram(unsigned int vertexShader, unsigned int pixelShader){

    unsigned int program = glCreateProgram();
    checkGlError("glCreateProgram");

    if (program) {

        glAttachShader(program, vertexShader);
        checkGlError("glAttachShader (vertexShader)");

        glAttachShader(program, pixelShader);
        checkGlError("glAttachShader (pixelShader)");

        glLinkProgram(program);

        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

        if (linkStatus != GL_TRUE) {
            GLint bufLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
            if (bufLength) {
                char* buf = (char*) malloc(bufLength);
                if (buf) {
                    glGetProgramInfoLog(program, bufLength, NULL, buf);
                    std::cerr << "oglShader:" << "Could not link program:" << std::endl << buf << std::endl << std::flush;
                    free(buf);
                }
            }
            glDeleteProgram(program);
            program = 0;
        }
    }
    return program;
}

GLuint oglShader::LoadShader(GLenum shaderType, const char* pSource) {

    GLint compiled = 0;
    GLint infoLen = 0;

    GLuint shader = glCreateShader(shaderType);
    checkGlError("glCreateShader");

    if (shader==0) return shader;

    GLSHADERSOURCE(shader, 1, &pSource, NULL);
    checkGlError("glShaderSource");

    GLCOMPILESHADER(shader);
    checkGlError("glCompileShader");

    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled==0) {
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        char* buf = (char*) malloc(infoLen);
        glGetShaderInfoLog(shader, infoLen, NULL, buf);
        std::cerr << "oglShader:" << "Could not compile shader" << shaderType << ":" << std::endl << buf << std::endl << std::flush;
        free(buf);
        glDeleteShader(shader);
        shader = 0;
    }

    return shader;
}


bool oglShader::Enable(){
    if( program == 0 ) return false;
    glUseProgram(program);
    checkGlError("glUseProgram");
    return true;
}

bool oglShader::Disable(){
    glUseProgram( 0 );
    checkGlError("glUseProgram");
    return true;
}

GLint oglShader::GetAttribLocation( const char * attr_name ){
    int loc = glGetAttribLocation( program, attr_name );
    if( loc == -1 ){
        std::cerr << "oglShader:" << "Attribute " << attr_name << " not found" << std::endl << std::flush;
    }
    return loc;
}

GLint oglShader::GetUniformLocation( const char * attr_name ){
    int loc = glGetUniformLocation( program, attr_name );
    if( loc == -1 ){
        std::cerr << "oglShader:" << "Uniform " << attr_name << " not found" << std::endl << std::flush;
    }
    return loc;
}

