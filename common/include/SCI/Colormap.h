/*
**  Common Library
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

#ifndef SCI_COLORMAP_H
#define SCI_COLORMAP_H

#include <vector>

#include <SCI/Vex4.h>
#include <SCI/Utility.h>

#include <GL/oglFont.h>

namespace SCI {
    class Colormap {

    public:
        Colormap( );

        void Insert( SCI::Vex4 c );
        void Insert( SCI::Vex3 c );

        virtual void Clear(){ cols.clear(); }

        virtual SCI::Vex4 GetColor( float t ) const = 0;
        int GetColorCount();

        virtual void Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font = 0 ) = 0;

    protected:
        std::vector< SCI::Vex4 > cols;
    };

    class SequentialColormap : public Colormap {
    public:
        SequentialColormap( bool strong = false );

        void LoadDefaultMapRed( );
        void LoadDefaultMapBlue( );
        void LoadDefaultMapRedBlueWhite( );

        void SetRange( float minv = 0, float maxv = 1 );
        float GetRangeMinimum(){ return minv; }
        float GetRangeMaximum(){ return maxv; }

        virtual void Clear(){ cols.clear(); minv = 0; maxv = 1; }

        virtual SCI::Vex4 GetColor( float t ) const ;

        virtual void Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font = 0 );

    protected:
        bool  strong;
        float minv;
        float maxv;

    };

    class CatagoricalColormap : public Colormap {
    public:
        CatagoricalColormap( );

        void LoadDefaultMap( );
        void LoadDefaultMapCohomology();

        virtual SCI::Vex4 GetColor( float t ) const ;

        virtual void Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font = 0 );

    };

    class DivergentColormap : public Colormap {
    public:
        DivergentColormap( bool strong = false );

        virtual SCI::Vex4 GetColor( float t ) const ;

        virtual void Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font = 0 );

    protected:
        bool strong;

    };

}

#endif // SCI_COLORMAP_H
