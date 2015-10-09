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

#ifndef CGPARAMETER_H
#define CGPARAMETER_H

class cgParameter {
protected:
    CGparameter param;
public:
    inline cgParameter(CGparameter _param = 0){ param = _param; }

    inline bool isValid(){ return (param != 0); }

    inline void   SetTextureParameter(GLuint texobj){    cgGLSetTextureParameter(param,texobj);    }
    inline GLuint GetTextureParameter(){                cgGLGetTextureParameter(param);            }
    inline void   EnableTextureParameter(){                cgGLEnableTextureParameter(param);        }
    inline void   DisableTextureParameter(){            cgGLDisableTextureParameter(param);        }
    inline GLenum GetTextureEnum(){                        cgGLGetTextureEnum(param);                }

    inline void SetParameter1f(float x){                                cgSetParameter1f(param, x);            }
    inline void SetParameter2f(float x, float y) {                        cgSetParameter2f(param, x,y);        }
    inline void SetParameter3f(float x, float y, float z) {                cgSetParameter3f(param, x,y,z);        }
    inline void SetParameter4f(float x, float y, float z, float w) {    cgSetParameter4f(param, x,y,z,w);    }

    inline void SetParameter1fv(const float * v){        cgSetParameter1fv(param, v);    }
    inline void SetParameter2fv(const float * v){        cgSetParameter2fv(param, v);    }
    inline void SetParameter3fv(const float * v){        cgSetParameter3fv(param, v);    }
    inline void SetParameter4fv(const float * v){        cgSetParameter4fv(param, v);    }

    inline void SetMatrixParameterdr(const double * matrix){    cgSetMatrixParameterdr(param, matrix);    }
    inline void SetMatrixParameterfr(const float  * matrix){    cgSetMatrixParameterfr(param, matrix);    }
    inline void SetMatrixParameterdc(const double * matrix){    cgSetMatrixParameterdc(param, matrix);    }
    inline void SetMatrixParameterfc(const float  * matrix){    cgSetMatrixParameterfc(param, matrix);    }

    inline void SetParameter1i(int x){                            int i[] = {x};            cgSetParameterValueir(param, 1, i);        }
    inline void SetParameter2i(int x, int y) {                    int i[] = {x,y};        cgSetParameterValueir(param, 2, i);        }
    inline void SetParameter3i(int x, int y, int z) {            int i[] = {x,y,z};        cgSetParameterValueir(param, 3, i);        }
    inline void SetParameter4i(int x, int y, int z, int w) {    int i[] = {x,y,z,w};    cgSetParameterValueir(param, 4, i);        }

    inline void SetParameter1iv(const int * v){        cgSetParameterValueir(param, 1, v);        }
    inline void SetParameter2iv(const int * v) {    cgSetParameterValueir(param, 2, v);        }
    inline void SetParameter3iv(const int * v) {    cgSetParameterValueir(param, 3, v);        }
    inline void SetParameter4iv(const int * v) {    cgSetParameterValueir(param, 4, v);        }

    inline float GetParameter1f(){                                                float tmp[1]; cgGetParameterValuefr(param, 1, tmp);    return tmp[0];                                    }
    inline void  GetParameter1f(float & x){                                        float tmp[1]; cgGetParameterValuefr(param, 1, tmp);    x = tmp[0];                                        }
    inline void  GetParameter2f(float & x, float & y) {                            float tmp[2]; cgGetParameterValuefr(param, 2, tmp);    x = tmp[0]; y = tmp[1];                            }
    inline void  GetParameter3f(float & x, float & y, float & z) {                float tmp[3]; cgGetParameterValuefr(param, 3, tmp);    x = tmp[0]; y = tmp[1]; z = tmp[2];                }
    inline void  GetParameter4f(float & x, float & y, float & z, float & w) {    float tmp[4]; cgGetParameterValuefr(param, 4, tmp);    x = tmp[0]; y = tmp[1]; z = tmp[2]; w = tmp[3]; }

    inline float * GetParameter1fv(float * v){        cgGetParameterValuefr(param, 1, v); return v;    }
    inline float * GetParameter2fv(float * v){        cgGetParameterValuefr(param, 2, v); return v;    }
    inline float * GetParameter3fv(float * v){        cgGetParameterValuefr(param, 3, v); return v;    }
    inline float * GetParameter4fv(float * v){        cgGetParameterValuefr(param, 4, v); return v;    }

    inline double * GetMatrixParameterdr(double * matrix){    cgGetMatrixParameterdr(param, matrix); return matrix;    }
    inline float  * GetMatrixParameterfr(float  * matrix){    cgGetMatrixParameterfr(param, matrix); return matrix;    }
    inline double * GetMatrixParameterdc(double * matrix){    cgGetMatrixParameterdc(param, matrix); return matrix;    }
    inline float  * GetMatrixParameterfc(float  * matrix){    cgGetMatrixParameterfc(param, matrix); return matrix;    }

    inline void SetParameterArray1f( long offset, long nelements, const float * v ){ cgGLSetParameterArray1f(param, offset, nelements, v); }
    inline void SetParameterArray2f( long offset, long nelements, const float * v ){ cgGLSetParameterArray2f(param, offset, nelements, v); }
    inline void SetParameterArray3f( long offset, long nelements, const float * v ){ cgGLSetParameterArray3f(param, offset, nelements, v); }
    inline void SetParameterArray4f( long offset, long nelements, const float * v ){ cgGLSetParameterArray4f(param, offset, nelements, v); }

    inline void SetMatrixParameterArraydr(long offset, long nelements, double * matrix){    cgGLSetMatrixParameterArraydr(param, offset, nelements, matrix);    }
    inline void SetMatrixParameterArrayfr(long offset, long nelements, float  * matrix){    cgGLSetMatrixParameterArrayfr(param, offset, nelements, matrix);    }
    inline void SetMatrixParameterArraydc(long offset, long nelements, double * matrix){    cgGLSetMatrixParameterArraydc(param, offset, nelements, matrix);    }
    inline void SetMatrixParameterArrayfc(long offset, long nelements, float  * matrix){    cgGLSetMatrixParameterArrayfc(param, offset, nelements, matrix);    }
};

#endif // CGPARAMETER_H
