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

#ifndef CGSHADER_H
#define CGSHADER_H

#include <cg/cg.h>
#include <cg/cggl.h>
#include <CG/cgParameter.h>

#pragma comment (lib,"cg.lib")
#pragma comment (lib,"cggl.lib")

///////////////////////////////////////////////////////////////////
/// \brief Interface class for the nVidia CG Shader Language 
///
/// This class provides an interface run programs written in the 
/// nVidia CG shader language.  The class must be constructed 
/// within an already created OpenGL context.  In addition, the 
/// shader will only work for the OpenGL context it was created 
/// within.
///////////////////////////////////////////////////////////////////
class cgShader  
{
protected:
    CGprogram    cgProgram[3];
    CGprofile    cgProfile[3];

    char fname[256];
    char * vpmain;
    char * gpmain;
    char * fpmain;

    char * code;

    bool loaded;

public:
    ///////////////////////////////////////////////////////////////
    /// \brief Constructor
    ///////////////////////////////////////////////////////////////
    cgShader();

    bool LoadProgram(const char * code, const char* vpmain = 0, const char* gpmain = 0, const char* fpmain = 0);

    bool Load(const char* fname, const char* vpmain = 0, const char* gpmain = 0, const char* fpmain = 0);

    bool Reload();

    bool Loaded();

    ///////////////////////////////////////////////////////////////
    /// \brief Deconstructor
    ///////////////////////////////////////////////////////////////
    virtual ~cgShader();

    ///////////////////////////////////////////////////////////////
    /// \brief Enables the execution of the program
    ///////////////////////////////////////////////////////////////
    void Enable();

    ///////////////////////////////////////////////////////////////
    /// \brief Disables the execution of the program
    ///////////////////////////////////////////////////////////////
    void Disable();

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the vertex
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    CGparameter GetVertexParameter(const char* name);

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the geometry
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    CGparameter GetGeometryParameter(const char* name);

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the fragment
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    CGparameter GetFragmentParameter(const char* name);


    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the vertex
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    cgParameter GetVertexParameter1(const char* name);

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the geometry
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    cgParameter GetGeometryParameter1(const char* name);

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the fragment
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    cgParameter GetFragmentParameter1(const char* name);
};


///////////////////////////////////////////////////////////////////
/// \brief Interface class for the nVidia CG Shader Language 
///
/// This class provides an interface run programs written in the 
/// nVidia CG shader language.  The class must be constructed 
/// within an already created OpenGL context.  In addition, the 
/// shader will only work for the OpenGL context it was created 
/// within.
///////////////////////////////////////////////////////////////////
class cgShader_3
{
protected:
    CGprogram    cgProgram[5];
    CGprofile    cgProfile[5];

    char fname[256];
    char * main[5];

    char * code;

    bool loaded;

    cgParameter __GetParameter(int pid, const char* name);

public:
    ///////////////////////////////////////////////////////////////
    /// \brief Constructor
    ///////////////////////////////////////////////////////////////
    cgShader_3();

    ///////////////////////////////////////////////////////////////
    /// \brief Deconstructor
    ///////////////////////////////////////////////////////////////
    virtual ~cgShader_3();


    bool LoadFromCode(const char * code, const char* vp_main = 0, const char* tc_main = 0, const char* te_main = 0, const char* gp_main = 0, const char* fp_main = 0);
    bool LoadFromFile(const char* fname, const char* vp_main = 0, const char* tc_main = 0, const char* te_main = 0, const char* gp_main = 0, const char* fp_main = 0);
    bool Reload();
    bool Loaded();


    ///////////////////////////////////////////////////////////////
    /// \brief Enables the execution of the program
    ///////////////////////////////////////////////////////////////
    void Enable();

    ///////////////////////////////////////////////////////////////
    /// \brief Disables the execution of the program
    ///////////////////////////////////////////////////////////////
    void Disable();

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the vertex
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    cgParameter GetVertexParameter(const char* name);

    cgParameter GetTessellationControlParameter(const char* name);
    cgParameter GetTessellationEvaluationParameter(const char* name);

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the geometry
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    cgParameter GetGeometryParameter(const char* name);

    ///////////////////////////////////////////////////////////////
    /// \brief Gets a CGparameter to set a variable in the fragment
    ///        shader execution
    ///
    /// \param name Variable name in the shader
    ///
    /// \return CGparameter of the variable
    ///////////////////////////////////////////////////////////////
    cgParameter GetFragmentParameter(const char* name);
};


#endif // CGSHADER_H
