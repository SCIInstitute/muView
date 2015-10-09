/*
**  Common CG Support Library
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


// cgShader.cpp: implementation of the cgShader class.
//
//////////////////////////////////////////////////////////////////////

#include <CG/cgShader.h>
//#include <CG/util.h>


#include <CG/cgInternal.h>
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#define SAFE_FREE(A) { if(A){ free(A); } A = 0; }
#define ERROR_MSG(A) { std::cerr << A << std::endl << std::flush; }

cgShader::cgShader(){
    cgProfile[0] = cgProfile[1] = cgProfile[2] = CG_PROFILE_UNKNOWN;
    cgProgram[0] = cgProgram[1] = cgProgram[2] = NULL;
    vpmain = gpmain = fpmain = 0;
    loaded = false;
    code = 0;
}

cgShader::~cgShader(){

    SAFE_FREE(code);

    SAFE_FREE(vpmain);
    SAFE_FREE(gpmain);
    SAFE_FREE(fpmain);

    for(int k = 0; k < 3; k++){
        if(cgProgram[k]){
            cgDestroyProgram(cgProgram[k]);
        }
    }
}

bool cgShader::LoadProgram(const char * _code, const char* _vpmain, const char* _gpmain, const char* _fpmain){

    strcpy(fname,"program");

    SAFE_FREE(code);
    SAFE_FREE(vpmain);
    SAFE_FREE(gpmain);
    SAFE_FREE(fpmain);

    if(_vpmain){ vpmain = _strdup(_vpmain); }
    if(_gpmain){ gpmain = _strdup(_gpmain); }
    if(_fpmain){ fpmain = _strdup(_fpmain); }

    code = _strdup(_code);

    return Reload();
}

bool cgShader::Load(const char* _fname, const char* _vpmain, const char* _gpmain, const char* _fpmain){

    strcpy_s(fname,256,_fname);

    SAFE_FREE(code);
    SAFE_FREE(vpmain);
    SAFE_FREE(gpmain);
    SAFE_FREE(fpmain);

    if(_vpmain){ vpmain = _strdup(_vpmain); }
    if(_gpmain){ gpmain = _strdup(_gpmain); }
    if(_fpmain){ fpmain = _strdup(_fpmain); }

    return Reload();
}

bool cgShader::Reload(){

    for(int k = 0; k < 3; k++){
        if(cgProgram[k]){
            cgDestroyProgram(cgProgram[k]);
        }
    }

    cgProfile[0] = cgProfile[1] = cgProfile[2] = CG_PROFILE_UNKNOWN;
    cgProgram[0] = cgProgram[1] = cgProgram[2] = NULL;

    bool ret = true;

    if(cgGetContext()){
        for(int i = 0; i < 3; i++){
            if(i == 0 && vpmain == 0){ continue; }
            if(i == 1 && gpmain == 0){ continue; }
            if(i == 2 && fpmain == 0){ continue; }

            switch(i){
                case 0: cgProfile[i] = cgGLGetLatestProfile(CG_GL_VERTEX);   break;
                case 1: cgProfile[i] = cgGLGetLatestProfile(CG_GL_GEOMETRY); break;
                case 2: cgProfile[i] = cgGLGetLatestProfile(CG_GL_FRAGMENT); break;
            }

            if (cgProfile[i] == CG_PROFILE_UNKNOWN){
                printf("Invalid profile type (");
                switch(i){
                    case 0: printf("CG_GL_VERTEX)\n");   break;
                    case 1: printf("CG_GL_GEOMETRY)\n"); break;
                    case 2: printf("CG_GL_FRAGMENT)\n"); break;
                }
                ret = false;
                continue;
            }

            cgGLSetOptimalOptions(cgProfile[i]);

            if(code){
                switch(i){
                    case 0: cgProgram[i] = cgCreateProgram(cgGetContext(), CG_SOURCE, code, cgProfile[i], vpmain, 0); break;
                    case 1: cgProgram[i] = cgCreateProgram(cgGetContext(), CG_SOURCE, code, cgProfile[i], gpmain, 0); break;
                    case 2: cgProgram[i] = cgCreateProgram(cgGetContext(), CG_SOURCE, code, cgProfile[i], fpmain, 0); break;
                }
            }
            else{
                switch(i){
                    case 0: cgProgram[i] = cgCreateProgramFromFile(cgGetContext(), CG_SOURCE, fname, cgProfile[i], vpmain, 0); break;
                    case 1: cgProgram[i] = cgCreateProgramFromFile(cgGetContext(), CG_SOURCE, fname, cgProfile[i], gpmain, 0); break;
                    case 2: cgProgram[i] = cgCreateProgramFromFile(cgGetContext(), CG_SOURCE, fname, cgProfile[i], fpmain, 0); break;
                }
            }

            if (cgProgram[i] == NULL){
                switch(i){
                    case 0: printf("CG ERROR (VERTEX) : %s\n",  cgGetErrorString(cgGetError())); break;
                    case 1: printf("CG ERROR (GEOMETRY) : %s\n",cgGetErrorString(cgGetError())); break;
                    case 2: printf("CG ERROR (FRAGMENT) : %s\n",cgGetErrorString(cgGetError())); break;
                }
                ret = false;
                continue;
            }

            cgGLLoadProgram(cgProgram[i]);
        }
    }

    if(loaded){
        if(ret){    printf("INFO (cgShader): Reloaded %s successfully\n",fname); }
        else{        printf("INFO (cgShader): Reloading %s failed\n",fname); }
    }
    else{
        if(ret){    printf("INFO (cgShader): Loaded %s successfully\n",fname); }
        else{        printf("INFO (cgShader): Loading %s failed\n",fname); }
    }

    loaded = true;

    return ret;
}


bool cgShader::Loaded(){
    return loaded;
}



void cgShader::Enable(){
    for(int k = 0; k < 3; k++){
        if(cgProgram[k]){
            cgGLEnableProfile(cgProfile[k]);
            cgGLBindProgram(cgProgram[k]);
        }
    }
}

void cgShader::Disable(){
    for(int k = 0; k < 3; k++){
        if(cgProgram[k]){
            cgGLUnbindProgram(cgProfile[k]);
            cgGLDisableProfile(cgProfile[k]);
        }
    }
}

CGparameter cgShader::GetVertexParameter(const char* name){
    CGparameter ret = 0;

    if(cgProgram[0] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[0], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return ret;
}

CGparameter cgShader::GetGeometryParameter(const char* name){
    CGparameter ret = 0;
    if(cgProgram[1] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[1], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return ret;
}

CGparameter cgShader::GetFragmentParameter(const char* name){
    CGparameter ret = 0;
    if(cgProgram[2] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[2], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return ret;
}

cgParameter cgShader::GetVertexParameter1(const char* name){
    CGparameter ret = 0;

    if(cgProgram[0] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[0], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return cgParameter(ret);
}

cgParameter cgShader::GetGeometryParameter1(const char* name){
    CGparameter ret = 0;
    if(cgProgram[1] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[1], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return cgParameter(ret);
}

cgParameter cgShader::GetFragmentParameter1(const char* name){
    CGparameter ret = 0;
    if(cgProgram[2] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[2], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return cgParameter(ret);
}
































cgShader_3::cgShader_3(){
    for(int i = 0; i < 5; i++){
        cgProfile[i] = CG_PROFILE_UNKNOWN;
        cgProgram[i] = NULL;
        main[i] = 0;
    }
    loaded = false;
    code = 0;
}

cgShader_3::~cgShader_3(){
    SAFE_FREE(code);
    for(int k = 0; k < 5; k++){
        SAFE_FREE(main[k]);
        if(cgProgram[k]){
            cgDestroyProgram(cgProgram[k]);
        }
    }
}

bool cgShader_3::LoadFromCode(const char * _code, const char* vp_main, const char* tc_main, const char* te_main, const char* gp_main, const char* fp_main){

    SAFE_FREE(code);
    code = _strdup(_code);
    return LoadFromFile("code",vp_main,tc_main,te_main,gp_main,fp_main);
}

bool cgShader_3::LoadFromFile(const char* _fname, const char* vp_main, const char* tc_main, const char* te_main, const char* gp_main, const char* fp_main){

    if( strcmp( _fname, "code" ) != 0 ){
        SAFE_FREE(code);
    }
    strcpy_s(fname,256,_fname);

    for(int k = 0; k < 5; k++){
        SAFE_FREE(main[k]);
    }

    if(vp_main){ main[0] = _strdup(vp_main); }
    if(tc_main){ main[1] = _strdup(tc_main); }
    if(te_main){ main[2] = _strdup(te_main); }
    if(gp_main){ main[3] = _strdup(gp_main); }
    if(fp_main){ main[4] = _strdup(fp_main); }

    return Reload();
}

bool cgShader_3::Reload(){

    for(int k = 0; k < 5; k++){
        if(cgProgram[k]){
            cgDestroyProgram(cgProgram[k]);
        }
        cgProfile[k] = CG_PROFILE_UNKNOWN;
        cgProgram[k] = NULL;
    }

    bool ret = true;

    if(cgGetContext()){
        for(int i = 0; i < 5; i++){
            if( main[i] == 0 ){ continue; }

            switch(i){
                case 0: cgProfile[i] = cgGLGetLatestProfile(CG_GL_VERTEX);                    break;
                case 1: cgProfile[i] = cgGLGetLatestProfile(CG_GL_TESSELLATION_CONTROL);    break;
                case 2: cgProfile[i] = cgGLGetLatestProfile(CG_GL_TESSELLATION_EVALUATION);    break;
                case 3: cgProfile[i] = cgGLGetLatestProfile(CG_GL_GEOMETRY);                break;
                case 4: cgProfile[i] = cgGLGetLatestProfile(CG_GL_FRAGMENT);                break;
            }

            if (cgProfile[i] == CG_PROFILE_UNKNOWN){
                printf("Invalid profile type (");
                switch(i){
                    case 0: printf("CG_GL_VERTEX)\n");                    break;
                    case 1: printf("CG_GL_TESSELLATION_CONTROL)\n");    break;
                    case 2: printf("CG_GL_TESSELLATION_EVALUATION)\n");    break;
                    case 3: printf("CG_GL_GEOMETRY)\n");                break;
                    case 4: printf("CG_GL_FRAGMENT)\n");                break;
                }
                ret = false;
                continue;
            }

            cgGLSetOptimalOptions(cgProfile[i]);

            if(code){
                cgProgram[i] = cgCreateProgram(cgGetContext(), CG_SOURCE, code, cgProfile[i], main[i], 0);
            }
            else{
                cgProgram[i] = cgCreateProgramFromFile(cgGetContext(), CG_SOURCE, fname, cgProfile[i], main[i], 0);
            }

            if (cgProgram[i] == NULL){
                switch(i){
                    case 0: printf("CG ERROR (VERTEX) : %s\n",   cgGetErrorString(cgGetError())); break;
                    case 1: printf("CG ERROR (TESS_CTRL) : %s\n",cgGetErrorString(cgGetError())); break;
                    case 2: printf("CG ERROR (TESS_EVAL) : %s\n",cgGetErrorString(cgGetError())); break;
                    case 3: printf("CG ERROR (GEOMETRY) : %s\n", cgGetErrorString(cgGetError())); break;
                    case 4: printf("CG ERROR (FRAGMENT) : %s\n", cgGetErrorString(cgGetError())); break;
                }
                ret = false;
                continue;
            }

            cgGLLoadProgram(cgProgram[i]);
        }
    }

    if(loaded){
        if(ret){    printf("INFO (cgShader): Reloaded %s successfully\n",fname); }
        else{        printf("INFO (cgShader): Reloading %s failed\n",fname); }
    }
    else{
        if(ret){    printf("INFO (cgShader): Loaded %s successfully\n",fname); }
        else{        printf("INFO (cgShader): Loading %s failed\n",fname); }
    }

    loaded = true;

    return ret;
}


bool cgShader_3::Loaded(){
    return loaded;
}



void cgShader_3::Enable(){
    for(int k = 0; k < 5; k++){
        if(cgProgram[k]){
            cgGLEnableProfile(cgProfile[k]);
            cgGLBindProgram(cgProgram[k]);
        }
    }
}

void cgShader_3::Disable(){
    for(int k = 0; k < 5; k++){
        if(cgProgram[k]){
            cgGLUnbindProgram(cgProfile[k]);
            cgGLDisableProfile(cgProfile[k]);
        }
    }
}


cgParameter cgShader_3::__GetParameter(int pid, const char* name){
    CGparameter ret = 0;

    if(cgProgram[pid] == 0){
        ERROR_MSG("No program specified!");
    }
    else{
        ret = cgGetNamedParameter(cgProgram[pid], name);
        if(ret == 0){
            ERROR_MSG("Parameter not found");
        }
    }
    return cgParameter(ret);
}


cgParameter cgShader_3::GetVertexParameter(const char* name){                    return __GetParameter(0,name);    }
cgParameter cgShader_3::GetTessellationControlParameter(const char* name){        return __GetParameter(1,name);    }
cgParameter cgShader_3::GetTessellationEvaluationParameter(const char* name){    return __GetParameter(2,name);    }
cgParameter cgShader_3::GetGeometryParameter(const char* name){                    return __GetParameter(3,name);    }
cgParameter cgShader_3::GetFragmentParameter(const char* name){                    return __GetParameter(4,name);    }


