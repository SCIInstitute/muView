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

#include <GL/oglDisplayListArray.h>

using namespace oglWidgets;

oglDisplayListArray::oglDisplayListArray(void){
    dl = 0xffffffff;
    list_cnt = 0;
}

oglDisplayListArray::~oglDisplayListArray(void){
    if(dl != 0xffffffff){
        glDeleteLists(dl,list_cnt);
    }
}

bool oglDisplayListArray::isValid(){
    return (dl != 0xffffffff);
}

void oglDisplayListArray::Init(int cnt){
    if(dl != 0xffffffff){
        glDeleteLists(dl,list_cnt);
    }
    dl = glGenLists(cnt);
    list_cnt = cnt;
}

void oglDisplayListArray::CompileBegin(int idx, bool execute){
    if(idx >= (int)list_cnt) return;

    if(execute){
        glNewList(dl+idx,GL_COMPILE_AND_EXECUTE);
    }
    else{
        glNewList(dl+idx,GL_COMPILE);
    }
}

void oglDisplayListArray::CompileEnd(){
    glEndList();
}

void oglDisplayListArray::Execute(int idx){
    glCallList(dl+idx);
}

void oglDisplayListArray::ExecuteAll(){
    GLuint * lists = new GLuint[list_cnt];
    for(int i = 0; i < (int)list_cnt; i++){
        lists[i] = dl+i;
    }
    glCallLists( list_cnt, GL_UNSIGNED_INT, lists );
    delete[] lists;
}
