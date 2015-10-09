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

#include <SCI/Colormap.h>
#include <GL/oglCommon.h>

using namespace SCI;

Colormap::Colormap( ) { }
void Colormap::Insert( SCI::Vex4 c ){ cols.push_back( c ); }
void Colormap::Insert( SCI::Vex3 c ){ cols.push_back( SCI::Vex4(c,1) ); }
int  Colormap::GetColorCount(){ return (int)cols.size(); }



SequentialColormap::SequentialColormap(bool _strong ) : strong(_strong), minv(0), maxv(1) { }
void SequentialColormap::SetRange(float _minv, float _maxv){ minv = _minv; maxv = _maxv; }
SCI::Vex4 SequentialColormap::GetColor( float v ) const {
    if( cols.size() == 0 ){ return SCI::Vex4(1,1,1,1); }

    float t = (v-minv)/(maxv-minv);
    if(strong && t <= 0.00001) return cols[0];

    float tf;
    if(strong) tf = 1.0f + SCI::Clamp(t,0.0f,1.0f)*(float)(cols.size()-2);
    else       tf = SCI::Clamp(t,0.0f,1.0f)*(float)(cols.size()-1);

    int   i0 = (int)tf;
    int   i1 = SCI::Min( i0+1, (int)cols.size()-1 );
    return SCI::lerp( cols[i0], cols[i1], tf-floorf(tf) );
}

void SequentialColormap::LoadDefaultMapRed( ){
    Clear();
    Insert( SCI::Vex4( 254, 229, 217, 10 ) / 255.0f );
    Insert( SCI::Vex4( 252, 187, 161, 20 ) / 255.0f );
    Insert( SCI::Vex4( 252, 146, 114, 30 ) / 255.0f );
    Insert( SCI::Vex4( 251, 106,  74, 40 ) / 255.0f );
    Insert( SCI::Vex4( 222,  45,  38, 50 ) / 255.0f );
    Insert( SCI::Vex4( 165,  15,  21, 60 ) / 255.0f );
}

void SequentialColormap::LoadDefaultMapBlue( ){
    Clear();
    Insert( SCI::Vex4( 242, 240, 247, 10 ) / 255.0f );
    Insert( SCI::Vex4( 218, 218, 235, 20 ) / 255.0f );
    Insert( SCI::Vex4( 188, 189, 220, 30 ) / 255.0f );
    Insert( SCI::Vex4( 158, 154, 200, 40 ) / 255.0f );
    Insert( SCI::Vex4( 117, 107, 177, 50 ) / 255.0f );
    Insert( SCI::Vex4(  84,  39, 143, 60 ) / 255.0f );
}

void SequentialColormap::LoadDefaultMapRedBlueWhite( ){
    Clear();
    Insert( SCI::Vex4( 255,   0,   0, 255 ) / 255.0f );
    Insert( SCI::Vex4(   0,   0, 255, 255 ) / 255.0f );
    Insert( SCI::Vex4( 255, 255, 255, 255 ) / 255.0f );
}

void SequentialColormap::Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font ){
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUAD_STRIP);
        for(int i = 0; i <= 30; i++){
            float t = SCI::lerp( GetRangeMinimum( ), GetRangeMaximum(), (float)i/30.0f );
            glColor4fv( GetColor( t ).data );
            glVertex3f( u_min, SCI::lerp( v_min, v_max, (float)i/30.0f ), 0.0f );
            glVertex3f( u_max, SCI::lerp( v_min, v_max, (float)i/30.0f ), 0.0f );
        }
    glEnd();

    glColor3f(0,0,0);
    glBegin(GL_LINE_LOOP);
        glVertex3f( u_min, v_min, 0.0f );
        glVertex3f( u_max, v_min, 0.0f );
        glVertex3f( u_max, v_max, 0.0f );
        glVertex3f( u_min, v_max, 0.0f );
    glEnd();

    if( font ){
        glColor3f(0,0,0);
        for(int i = 0; i <= 10; i+=2){
            glPushMatrix();
                float t = SCI::lerp( GetRangeMinimum( ), GetRangeMaximum(), (float)i/10.0f );
                glTranslatef(u_max+0.005f, SCI::lerp( v_min, v_max, (float)i/10.0f ), 0.0f);
                glScalef(0.05f,0.05f*window_aspect,0.5f);
                glTranslatef(0, SCI::lerp( 0.1f, -0.8f, (float)i/10.0f), 0);
                font->Printf( "%1.2f", t );
            glPopMatrix();
        }
    }
}


CatagoricalColormap::CatagoricalColormap( ){ }
SCI::Vex4 CatagoricalColormap::GetColor( float t ) const { while(t<0){t+=cols.size();} return cols[ ((int)t)%cols.size() ]; }
void CatagoricalColormap::LoadDefaultMap(){
    Clear();
    Insert( SCI::Vex4( 141, 211, 199, 15 ) / 255.0f );
    Insert( SCI::Vex4( 255, 255, 179, 15 ) / 255.0f );
    Insert( SCI::Vex4( 190, 186, 218, 15 ) / 255.0f );
    Insert( SCI::Vex4( 251, 128, 114, 15 ) / 255.0f );
    Insert( SCI::Vex4( 128, 177, 211, 15 ) / 255.0f );
    Insert( SCI::Vex4( 253, 180,  98, 15 ) / 255.0f );
    Insert( SCI::Vex4( 179, 222, 105, 15 ) / 255.0f );
    Insert( SCI::Vex4( 252, 205, 229, 15 ) / 255.0f );
    Insert( SCI::Vex4( 217, 217, 217, 15 ) / 255.0f );
    Insert( SCI::Vex4( 188, 128, 189, 15 ) / 255.0f );
    Insert( SCI::Vex4( 204, 235, 197, 15 ) / 255.0f );
    Insert( SCI::Vex4( 255, 237, 111, 15 ) / 255.0f );
}

void CatagoricalColormap::LoadDefaultMapCohomology(){
    Clear();
    Insert( SCI::Vex4(141,211,199,255) / 255 );
    Insert( SCI::Vex4(255,255,179,255) / 255 );
    Insert( SCI::Vex4(190,186,218,255) / 255 );
    Insert( SCI::Vex4(251,128,114,255) / 255 );
    Insert( SCI::Vex4(128,177,211,255) / 255 );
    Insert( SCI::Vex4(253,180, 98,255) / 255 );
    Insert( SCI::Vex4(179,222,105,255) / 255 );
    Insert( SCI::Vex4(252,205,229,255) / 255 );
    Insert( SCI::Vex4(188,128,189,255) / 255 );
    Insert( SCI::Vex4(102,194,165,255) / 255 );
    Insert( SCI::Vex4(141,160,203,255) / 255 );
    Insert( SCI::Vex4(255,217, 47,255) / 255 );
    Insert( SCI::Vex4(166,216, 84,255) / 255 );
    Insert( SCI::Vex4(231,138,195,255) / 255 );
}

void CatagoricalColormap::Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font ){
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUAD_STRIP);
        for(int i = 0; i < (int)cols.size(); i++){
            glColor4fv( GetColor( i ).data );
            glVertex3f( u_min, SCI::lerp( v_min, v_max, (float)i/(float)cols.size() ), 0.0f );
            glVertex3f( u_max, SCI::lerp( v_min, v_max, (float)i/(float)cols.size() ), 0.0f );
        }
    glEnd();

    glColor3f(0,0,0);
    glBegin(GL_LINE_LOOP);
        glVertex3f( u_min, v_min, 0.0f );
        glVertex3f( u_max, v_min, 0.0f );
        glVertex3f( u_max, v_max, 0.0f );
        glVertex3f( u_min, v_max, 0.0f );
    glEnd();

    /*
    if( font ){
        for(int i = 0; i <= 10; i+=2){
            glPushMatrix();
                float t = SCI::lerp( GetRangeMinimum( ), GetRangeMaximum(), (float)i/10.0f );
                glTranslatef(u_max+0.005f, SCI::lerp( v_min, v_max, (float)i/10.0f ), 0.0f);
                glScalef(0.03f,0.03f*window_aspect,1.0f);
                glTranslatef(0, SCI::lerp( 0.1f, -0.8f, (float)i/10.0f), 0);
                font->Printf( "%f", t );
            glPopMatrix();
        }
    }
    */
}

DivergentColormap::DivergentColormap( bool _strong ) : strong(_strong){ }

SCI::Vex4 DivergentColormap::GetColor( float t ) const {
    t = SCI::Clamp( t, -1.0f, 1.0f );

    float r = (float)(cols.size()/2);

    if(  strong && t <  0 ){ r = SCI::lerp( (float)(cols.size()/2-1),                          0, -t ); }
    if(  strong && t >  0 ){ r = SCI::lerp( (float)(cols.size()/2+1), (float)cols.size()-1.0001f,  t ); }
    if( !strong && t <= 0 ){ r = SCI::lerp( (float)(cols.size()/2),                            0, -t ); }
    if( !strong && t >= 0 ){ r = SCI::lerp( (float)(cols.size()/2),   (float)cols.size()-1.0001f,  t ); }

    return SCI::lerp( cols[(int)r], cols[(int)r+1], r-floorf(r) );
}


void DivergentColormap::Draw( float u_min, float u_max, float v_min, float v_max, float window_aspect, oglWidgets::oglFont * font ){
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUAD_STRIP);
        for(int i = 0; i <= 30; i++){
            float t = SCI::lerp( -1.0f, 1.0f, (float)i/30.0f );
            glColor4fv( GetColor( t ).data );
            glVertex3f( u_min, SCI::lerp( v_min, v_max, (float)i/30.0f ), 0.0f );
            glVertex3f( u_max, SCI::lerp( v_min, v_max, (float)i/30.0f ), 0.0f );
        }
    glEnd();

    glColor3f(0,0,0);
    glBegin(GL_LINE_LOOP);
        glVertex3f( u_min, v_min, 0.0f );
        glVertex3f( u_max, v_min, 0.0f );
        glVertex3f( u_max, v_max, 0.0f );
        glVertex3f( u_min, v_max, 0.0f );
    glEnd();

    if( font ){
        for(int i = 0; i <= 10; i+=2){
            glPushMatrix();
                float t = SCI::lerp( -1.0f, 1.0f, (float)i/10.0f );
                glTranslatef(u_max+0.005f, SCI::lerp( v_min, v_max, (float)i/10.0f ), 0.0f);
                glScalef(0.03f,0.03f*window_aspect,1.0f);
                glTranslatef(0, SCI::lerp( 0.1f, -0.8f, (float)i/10.0f), 0);
                font->Printf( "%f", t );
            glPopMatrix();
        }
    }
}
