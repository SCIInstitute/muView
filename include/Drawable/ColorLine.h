#ifndef COLORLINE_H
#define COLORLINE_H

#include <SCI/Vex2.h>
#include <SCI/Vex4.h>
#include <MyLib/PolylineSimplification.h>

#include <vector>

#include <QColor>

namespace Drawable {

    class ColorLine {
    public:
        ColorLine();

        void Clear();
        void AddPoint( SCI::Vex2 p );
        void Draw( );
        void DrawSelected( );

        float GetParameterization( SCI::Vex2 p );

        float DistanceToHead( SCI::Vex2 p );
        float DistanceToTail( SCI::Vex2 p );

        void SelectClosestControlPoint( SCI::Vex2 p );

        void SelectedControlPointDialog( );

        SCI::Vex4 GetColor( SCI::Vex2 p );

    protected:
        int selected;
        std::vector< SCI::Vex3 >       input;
        MyLib::PolylineSimplificationRDP simplified;
        float                            total_length;
        std::vector< SCI::Vex4 >       colors;

    };

}


#endif // COLORLINE_H
