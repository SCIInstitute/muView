#ifndef MeshDisplay_H
#define MeshDisplay_H

#include <SCI/Subset.h>
#include <Data/BasicData.h>

#include <Drawable/DrawableObject.h>
#include <Drawable/Provenance.h>

#include <SCI/Camera/FrustumProjection.h>
#include <SCI/Camera/ThirdPersonCameraControls.h>

#include <GL/oglTexture3D.h>
#include <GL/oglShader.h>

namespace Drawable {

    // Class for rendering volume data
    class MeshDisplay : public DrawableObject
    {
        Q_OBJECT

    public:
        MeshDisplay();

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

        SCI::FrustumProjection           proj;
        SCI::ThirdPersonCameraControls   view;

        unsigned int                     * vol_data;
        Drawable::ProvenanceNode         * curr_node;
        bool updated;

        SCI::Vex3 bbmin, bbmax;

        unsigned int * col_data;

        unsigned int * ClearColor();

    };
}

#endif // MeshDisplay_H
