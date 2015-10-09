#ifndef POINTDISPLAY_H
#define POINTDISPLAY_H


#if 0
#include <SCI/Subset.h>

#include <Data/BasicData.h>
#include <Data/PointData.h>

#include <Drawable/DrawableObject.h>

#include <MyLib/FrustumProjection.h>
#include <MyLib/ThirdPersonCameraControls.h>

#include <GL/oglTexture3D.h>
#include <GL/oglShader.h>

namespace Drawable {

    // Class for rendering volume data
    class HeartPointDisplay : public DrawableObject
    {
        Q_OBJECT

    public:
        HeartPointDisplay();

        void Recalculate( int dim = -1 );

        // Draw the data
        virtual void Draw();

        // Sets the data
        void Initialize( Data::PointData & _data );

        virtual bool MouseClick(int button, int state, int x, int y);
        virtual bool MouseMotion(int button, int x, int dx, int y, int dy);
        virtual bool Keypress( int key, int x, int y );

    public slots:
        void ResetView();

    protected:

        bool mouse_active;

        Data::PointData                  * data;

        MyLib::FrustumProjection           proj;
        MyLib::ThirdPersonCameraControls   view;

        bool updated;

        SCI::Vex3 bbmin, bbmax;

        unsigned int * col_data;

        unsigned int * ClearColor();

        int curdim;

    };
}

#endif

#endif // POINTDISPLAY_H
