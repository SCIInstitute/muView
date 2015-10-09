#ifndef DIMREDDISPLAY_H
#define DIMREDDISPLAY_H

#include <SCI/Vex2.h>
#include <SCI/Mat4.h>
#include <SCI/Timer.h>

#include <Data/VolData.h>

#include <DimensionalityReduction.h>

#include <Drawable/DrawableObject.h>
#include <Drawable/Provenance.h>
#include <Drawable/BarChart.h>
#include <Drawable/ColorLine.h>

#include <GL/oglFrameBufferObject.h>

namespace Drawable {

    // Class for drawing dimensional reduction data
    class DimRedDisplay : public DrawableObject
    {
    public:
        DimRedDisplay();

        // Function for initializing the data
        void Initialize( Provenance & _prov, Data::BasicData &_data );

        // Drawing function
        virtual void Draw( );

        virtual bool MouseClick(int button, int state, int x, int y);
        virtual bool MouseMotion(int button, int x, int dx, int y, int dy);

        //int GetClosestVoxelID( int mouse_x, int mouse_y );
        //std::vector<int> GetVoxelsInRadius( int mouse_x, int mouse_y, float radius );
        int UpdateAverageInRadius( );
        int UpdateAverageInBBox( );

    protected:
        bool mouse_active;

        Provenance * prov;

        SCI::Vex2 sel_min, sel_max;
        SCI::Vex2 mp;

        Drawable::BarChart      draw_bar;

        SCI::Timer mouse_hold_timer;
        int          selected_voxel_id;
        int mouse_x, mouse_y;

        //std::vector< SCI::Vex2 > line;
        bool line_active;

    };
}

#endif // DIMREDDISPLAY_H
