#ifndef VOLUMERENDERER_H
#define VOLUMERENDERER_H

#include <GL/oglTexture3D.h>

#include <SCI/Mat4.h>

class VolumeRenderer {

public:
    VolumeRenderer();

    // Draw the data
    void Draw(SCI::Mat4 mvp);

    // Sets the data
    void Initialize( int dim );

    int GetDim(){ return dim; }

    void SetSize( SCI::Vex3 center, SCI::Vex3 scale );

    unsigned int * ClearVolume( );
    unsigned int * ClearVolume( unsigned int col );

    int GetVoxelID( int x, int y, int z ){ return z*dim*dim + y*dim + x; }

    SCI::Vex3 VoxelCenter( int x, int y, int z );
    void SetVoxel( int x, int y, int z, SCI::Vex4 col );

    void ClipX( bool enable, float w );
    void ClipY( bool enable, float w );
    void ClipZ( bool enable, float w );

    void UpdateVolume( );

    bool isInitialized();

protected:

    SCI::Vex3 center,scale;
    oglWidgets::oglTexture3D           vol_tex;

    unsigned int                     * vol_data;
    int                                dim;
    bool                               need_update;

    void Vertex( SCI::Mat4 & imvp, SCI::Vex3 pnt, SCI::Vex3 volmin, SCI::Vex3 volmax );


};

#endif // VOLUMERENDERER_H
