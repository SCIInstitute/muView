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

#include <CG/cgInternal.h>

#include <stdio.h>

CGcontext __cgContext = NULL;

void cgErrorCallback(){
    CGerror lastError = cgGetError();
    if(lastError)
    {
        printf("%s\n", cgGetErrorString(lastError));
        //printf("%s\n", cgGetLastListing(gContext));
        printf("%s\n", cgGetLastListing(cgGetContext()));
        //exit(1);
    }
}

CGcontext cgGetContext(){
    if(__cgContext != NULL){
        return __cgContext;
    }

    if ((__cgContext = cgCreateContext()) == NULL){
        printf("Failed To Create Cg Context\n");
        return 0;
    }

    cgSetErrorCallback(cgErrorCallback);

    cgGLRegisterStates(cgGetContext());

    return __cgContext;
}


void cgDestroyContext(){
    if(__cgContext){
        cgDestroyContext(__cgContext);
        __cgContext = 0;
    }
}

