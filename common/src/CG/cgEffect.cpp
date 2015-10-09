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

#include <CG/cgEffect.h>
//#include <mylib/util.h>

#include <stdio.h>
#include <string.h>

#include <stdio.h>

#include <CG/cgInternal.h>
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


cgEffect::cgEffect(void){
    gEffect = 0;
    for(int i = 0; i < 32; i++){
        techniques[i] = 0;
    }
}

cgEffect::~cgEffect(void){
    if(gEffect){
        cgDestroyEffect(gEffect);
    }
}

bool cgEffect::LoadFromString(const char * str){
    gEffect = cgCreateEffect(cgGetContext(), str, 0);

    if(!gEffect){
        printf("Could not load from string\n");
    }

    techniques[0] = cgGetFirstTechnique(gEffect);
    if(techniques[0]){
        if(cgValidateTechnique(techniques[0]) == CG_TRUE){
            printf("Loaded technique '%s'\n",cgGetTechniqueName(techniques[0]));
        }
        else{
            printf("Technique '%s' did not validate\n",cgGetTechniqueName(techniques[0]));
        }
    }

    for(int i = 1; techniques[i-1] && i < 32; i++){
        techniques[i] = cgGetNextTechnique(techniques[i-1]);
        //printf("Loaded technique '%s'\n",cgGetTechniqueName(techniques[i]));

        if(techniques[i]){
            if(cgValidateTechnique(techniques[i]) == CG_TRUE){
                printf("Loaded technique '%s'\n",cgGetTechniqueName(techniques[i]));
            }
            else{
                printf("Technique '%s' did not validate\n",cgGetTechniqueName(techniques[i]));
            }
        }

    }

    return true;
}

bool cgEffect::Load(const char * fname){
    gEffect = cgCreateEffectFromFile(cgGetContext(), fname, 0);

    if(!gEffect){
        printf("Could not load '%s'\n",fname);
    }

    techniques[0] = cgGetFirstTechnique(gEffect);
    if(techniques[0]){
        if(cgValidateTechnique(techniques[0]) == CG_TRUE){
            printf("Loaded technique '%s'\n",cgGetTechniqueName(techniques[0]));
        }
        else{
            printf("Technique '%s' did not validate\n",cgGetTechniqueName(techniques[0]));
        }
    }

    for(int i = 1; techniques[i-1] && i < 32; i++){
        techniques[i] = cgGetNextTechnique(techniques[i-1]);
        //printf("Loaded technique '%s'\n",cgGetTechniqueName(techniques[i]));

        if(techniques[i]){
            if(cgValidateTechnique(techniques[i]) == CG_TRUE){
                printf("Loaded technique '%s'\n",cgGetTechniqueName(techniques[i]));
            }
            else{
                printf("Technique '%s' did not validate\n",cgGetTechniqueName(techniques[i]));
            }
        }

    }

    return true;
}

CGtechnique cgEffect::GetTechnique(const char * name){
    for(int i = 0; i < 32; i++){
        if(techniques[i]){
            if(strcmp(name,cgGetTechniqueName(techniques[i])) == 0){
                return techniques[i];
            }
        }
    }
    printf("Could not find technique '%s'\n",name);
    return 0;
}

/*
CGparameter cgEffect::GetParameter(const char* name ){

    CGparameter ret = cgGetEffectParameterBySemantic(gEffect, name);
    if(!ret){
        printf("getCGParam: Couldn't find parameter with %s semantic\n",name);
    }
    return ret;
}
*/

cgParameter cgEffect::GetParameter(const char * name){
    CGparameter ret = cgGetEffectParameterBySemantic(gEffect, name);
    if(!ret){
        printf("getCGParam: Couldn't find parameter with %s semantic\n",name);
    }
    return cgParameter(ret);
}

