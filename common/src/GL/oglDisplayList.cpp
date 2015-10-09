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

#include <GL/oglDisplayList.h>

using namespace oglWidgets;

oglDisplayList::oglDisplayList(void){
    dl = 0xffffffff;
}

oglDisplayList::~oglDisplayList(void){
    if(dl != 0xffffffff){
        glDeleteLists(dl,1);
    }
}

bool oglDisplayList::isValid(){
    return (dl != 0xffffffff);
}

void oglDisplayList::CompileBegin(bool execute){
    if(dl == 0xffffffff){
        dl = glGenLists(1);
    }

    if(execute){
        glNewList(dl,GL_COMPILE_AND_EXECUTE);
    }
    else{
        glNewList(dl,GL_COMPILE);
    }
}

void oglDisplayList::CompileEnd(){
    glEndList();
}

void oglDisplayList::Execute(){
    glCallList(dl);
}
