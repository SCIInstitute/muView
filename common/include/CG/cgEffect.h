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

#ifndef CGEFFECT_H
#define CGEFFECT_H

#include <cg/cg.h>
#include <cg/cggl.h>
#include <CG/cgParameter.h>

#pragma comment (lib,"cg.lib")
#pragma comment (lib,"cggl.lib")

class cgEffect {

public:
    CGeffect    gEffect;
    CGtechnique techniques[32];

public:
    cgEffect(void);
    ~cgEffect(void);

    bool Load(const char * fname);
    bool LoadFromString(const char * str);

    CGtechnique GetTechnique(const char * name);
    cgParameter GetParameter(const char * name);

};


#ifndef CG_RUN_TECHNIQUE
    #define CG_RUN_TECHNIQUE( TECHNIQUE, DRAW_FUNC ) {                                            \
                                                         CGtechnique    tech = TECHNIQUE;         \
                                                         CGpass lPass     = cgGetFirstPass(tech); \
                                                         while (lPass) {                          \
                                                             cgSetPassState(lPass);               \
                                                             DRAW_FUNC;                           \
                                                             cgResetPassState(lPass);             \
                                                             lPass = cgGetNextPass(lPass);        \
                                                         }                                        \
                                                     }
#endif

#endif // CGEFFECT_H
