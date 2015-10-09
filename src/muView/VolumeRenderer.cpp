#include <muView/VolumeRenderer.h>


VolumeRenderer::VolumeRenderer(){
    vol_data     = 0;
    center.Set(0,0,0);
    scale.Set(1,1,1);
    need_update = false;
}

void VolumeRenderer::Initialize( int _dim ){
    dim = _dim;
    vol_data = new unsigned int[ dim*dim*dim ];
    need_update = true;
}

bool VolumeRenderer::isInitialized(){
    return vol_data != 0;
}

void VolumeRenderer::SetSize( SCI::Vex3 _center, SCI::Vex3 _scale ){
    center = _center;
    scale  = _scale;
}

void VolumeRenderer::Vertex( SCI::Mat4 & imvp, SCI::Vex3 pnt, SCI::Vex3 volmin, SCI::Vex3 volmax ){
    SCI::Vex3 p = imvp * pnt;
    SCI::Vex3 t = (p-volmin)/(volmax-volmin);
    glTexCoord3fv(t.data);
    glVertex3fv(p.data);
}

unsigned int * VolumeRenderer::ClearVolume( ){
    return (unsigned int*)memset( vol_data, 0, sizeof(int)*dim*dim*dim );
}

unsigned int * VolumeRenderer::ClearVolume( unsigned int col ){
    for(int i = 0; i < dim*dim*dim; i++ ){
        vol_data[i] = col;
    }
    return vol_data;
}

void VolumeRenderer::UpdateVolume( ){
    need_update = true;
}

SCI::Vex3 VolumeRenderer::VoxelCenter( int x, int y, int z ){

    SCI::Vex3 m0 = center - scale;
    SCI::Vex3 m1 = center + scale;
    SCI::Vex3 ret;

    ret.x = SCI::lerp( m0.x, m1.x, ((float)x+0.5f)/dim );
    ret.y = SCI::lerp( m0.y, m1.y, ((float)y+0.5f)/dim );
    ret.z = SCI::lerp( m0.z, m1.z, ((float)z+0.5f)/dim );

    return ret;
}

void VolumeRenderer::SetVoxel( int x, int y, int z, SCI::Vex4 col ){
    vol_data[ z*dim*dim + y*dim + x ] = col.UIntColor();
}

void VolumeRenderer::Draw( SCI::Mat4 mvp ){
    SCI::Mat4 imvp = mvp.Inverse();

    if(need_update){
        vol_tex.SetMinMagFilter( GL_LINEAR, GL_LINEAR );
        vol_tex.SetWrapR( GL_CLAMP );
        vol_tex.SetWrapS( GL_CLAMP );
        vol_tex.SetWrapT( GL_CLAMP );
        vol_tex.TexImage3D( GL_RGBA, dim, dim, dim, GL_RGBA, GL_UNSIGNED_BYTE, vol_data );
        need_update = false;
    }

    SCI::Vex3 minmax[2];
    minmax[0] = center - scale;
    minmax[1] = center + scale;

    SCI::Vex3 minp = SCI::VEX3_MAX;
    SCI::Vex3 maxp = SCI::VEX3_MIN;
    for(int i = 0; i < 8; i++){
        SCI::Vex3 outp = mvp * SCI::Vex3( minmax[i%2].x, minmax[(i/2)%2].y, minmax[(i/4)%2].z );;
        minp = SCI::Min( minp, outp );
        maxp = SCI::Max( maxp, outp );
    }
    minp = SCI::Max( minp, SCI::Vex3(-1,-1,-1) );
    maxp = SCI::Min( maxp, SCI::Vex3( 1, 1, 1) );

    glDisable(GL_DEPTH_TEST);

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    int slices = 200;
    vol_tex.Enable();
    vol_tex.Bind();
        glColor3f(1,1,1);
        glBegin(GL_QUADS);
        for(int i = slices-1; i>=0; i--){
            float curz = SCI::lerp( minp.z, maxp.z, (float)i/(float)(slices-1) );
            Vertex( imvp, SCI::Vex3( minp.x, minp.y, curz ), minmax[0], minmax[1] );
            Vertex( imvp, SCI::Vex3( maxp.x, minp.y, curz ), minmax[0], minmax[1] );
            Vertex( imvp, SCI::Vex3( maxp.x, maxp.y, curz ), minmax[0], minmax[1] );
            Vertex( imvp, SCI::Vex3( minp.x, maxp.y, curz ), minmax[0], minmax[1] );
        }
        glEnd();
    vol_tex.Disable();

    glDisable( GL_BLEND );
}



