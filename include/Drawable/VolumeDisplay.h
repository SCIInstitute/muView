#ifndef VOLUMEDISPLAY_H
#define VOLUMEDISPLAY_H

#include <SCI/Subset.h>
#include <Data/BasicData.h>

#include <Drawable/DrawableObject.h>
#include <Drawable/Provenance.h>

#include <MyLib/FrustumProjection.h>
#include <MyLib/ThirdPersonCameraControls.h>

#include <GL/oglTexture3D.h>
#include <GL/oglShader.h>

namespace Drawable {

    // Class for rendering volume data
    class VolumeDisplay : public DrawableObject
    {
        Q_OBJECT

    public:
        VolumeDisplay();

        // Draw the data
        virtual void Draw();

        // Sets the data
        void Initialize( Data::ProxyData & _data,  Drawable::Provenance & _prov );

        virtual bool MouseClick(int button, int state, int x, int y);
        virtual bool MouseMotion(int button, int x, int dx, int y, int dy);

    public slots:
        void ResetView();

    protected:

        bool mouse_active;

        Data::ProxyData                  * data;
        Drawable::Provenance             * prov;

        MyLib::FrustumProjection           proj;
        MyLib::ThirdPersonCameraControls   view;

        oglWidgets::oglTexture3D           vol_tex;

        unsigned int                     * vol_data;
        Drawable::ProvenanceNode         * curr_node;
        bool updated;

    protected:
        class VolumeShader : public oglWidgets::oglShader {
        public:
            VolumeShader();
            void Load( );
        };

        VolumeShader                       shader;

    protected:
        unsigned int * ClearVolume( );
        void UpdateVolume( );
        void Vertex( SCI::Mat4 & imvp, SCI::Vex3 pnt, SCI::Vex3 volmin, SCI::Vex3 volmax );

    };

}

#endif // VOLUMEDISPLAY_H
