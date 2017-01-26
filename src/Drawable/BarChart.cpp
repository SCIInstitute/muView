#include <Drawable/BarChart.h>

#include <GL/oglCommon.h>
#include <math.h>
#include <float.h>

using namespace Drawable;

BarChart::BarChart(){
    vmin = 0;
    vmax = 1;
}

void BarChart::SetMinimumValue( float v ){
    vmin = v;
}

void BarChart::SetMaximumValue( float v ){
    vmax = v;
}


void BarChart::Draw( ){
    int dimN = avg.size();

    if( dimN > 20 ){
        std::vector<float> newsum(20,0);
        std::vector<int>   newcnt(20,0);
        std::vector<float> newavg(20,0);
        std::vector<float> newmin(20,FLT_MAX);
        std::vector<float> newmax(20,FLT_MIN);

        for(int i = 0; i < dimN; i++){
            int bin = i*20/dimN;
            newcnt[bin] += 1;
            newsum[bin] += avg[i];
            newmin[bin]  = fmin(newmin[bin],minval[i]);
            newmax[bin]  = fmax(newmax[bin],maxval[i]);
        }

        for(int i = 0; i < 20; i++){
            if(newcnt[i]>0)
                newavg[i] = newsum[i] / (float)newcnt[i];
        }

        BarChart bc;
        bc.SetAverage(newavg);
        bc.SetMinimum(newmin);
        bc.SetMaximum(newmax);
        bc.SetShape( u0, v0, width, height );
        bc.Draw();
        return;
    }


    SetViewport();

    Clear();

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( -0.2f, (float)dimN+0.2f, vmin-(vmax-vmin)*0.05f, vmax+(vmax-vmin)*0.05f, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glColor3f(0.6f,0.6f,0.6f);
    glBegin(GL_QUADS);
    for(int i = 0; i < dimN; i++){
        float v = fmax( avg[i], 0.02f );
        glVertex3f( (float)i+0.15f, 0, 0 );
        glVertex3f( (float)i+0.85f, 0, 0 );
        glVertex3f( (float)i+0.85f, v, 0 );
        glVertex3f( (float)i+0.15f, v, 0 );
    }
    glEnd();

    glLineWidth(1.0f);
    glColor3f(0.2f,0.2f,0.2f);
    for(int i = 0; i < dimN; i++){
        float v = fmax( avg[i], 0.02f );
        glBegin(GL_LINE_LOOP);
            glVertex3f( (float)i+0.15f, 0, 0 );
            glVertex3f( (float)i+0.85f, 0, 0 );
            glVertex3f( (float)i+0.85f, v, 0 );
            glVertex3f( (float)i+0.15f, v, 0 );
        glEnd();
    }

    glLineWidth(2.0f);
    glColor3f(0.5f,0.0f,0.0f);
    for(int i = 0; i < dimN; i++){
        float v0 = fmax( minval[i], 0.02f );
        float v1 = fmax( maxval[i], 0.02f );
        glBegin(GL_LINES);
            glVertex3f( (float)i+0.15f, v0, 0 );
            glVertex3f( (float)i+0.85f, v0, 0 );
            glVertex3f( (float)i+0.15f, v1, 0 );
            glVertex3f( (float)i+0.85f, v1, 0 );
            glVertex3f( (float)i+0.50f, v0, 0 );
            glVertex3f( (float)i+0.50f, v1, 0 );
        glEnd();
    }

    glBegin(GL_LINES);
        glVertex3f( -0.2f,0,0 );
        glVertex3f( (float)dimN+0.2f, 0,0 );
    glEnd();

    DrawFrame();

}

void BarChart::SetAverage( std::vector<float> vals ){
    avg    = vals;
}

void BarChart::SetMinimum( std::vector<float> vals ){
    maxval = vals;
}

void BarChart::SetMaximum( std::vector<float> vals ){
    minval = vals;
}
