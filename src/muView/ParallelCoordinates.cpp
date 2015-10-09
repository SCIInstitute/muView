#include <muView/ParallelCoordinates.h>
#include <GL/oglCommon.h>

#include <QMouseEvent>
#include <iostream>

ParallelCoordinates::ParallelCoordinates( QWidget * parent ) : QGLWidget( QGLFormat(QGL::SingleBuffer | QGL::DepthBuffer | QGL::Rgba | QGL::AlphaChannel | QGL::DirectRendering | QGL::SampleBuffers), parent ){
    pdata    = 0;
    color    = 0;
    curDraw  = 0;
    selected = -1;
}

void ParallelCoordinates::SetData(Data::MultiDimensionalData *_pdata, Data::Mesh::ColorMesh * _cm ){
    if( pdata == _pdata && color == _cm ) return;

    pdata = _pdata;
    color = _cm;

    if(pdata != 0 && color != 0){

        SCI::BoundingRange br;

        for(int i = 0; i < pdata->GetElementCount(); i++){
            for(int d = 0; d < pdata->GetDimension(); d++){
                br.Expand( pdata->GetElement(i,d) );
            }
        }

        minv = br.GetCenter() - br.GetSize()*1.01f/2.0f;
        maxv = br.GetCenter() + br.GetSize()*1.01f/2.0f;
        dim  = pdata->GetDimension();

        dimLoc.clear();
        divLoc.clear();

        for(int i = 0; i < dim; i++){
            float loc = SCI::lerp( -1.0f, 1.0f, ((float)i+0.5f) / ((float)dim) );
            dimLoc.push_back( std::make_pair(loc,i) );
        }
        for(int i = 0; i < dim-1; i++){
            divLoc.push_back( SCI::lerp( -1.0f, 1.0f, ((float)i+0.5f) / ((float)dim-1.0f) ) );
        }

        Reset();
    }
}

void ParallelCoordinates::Reset(){
    curDraw = 0;
}

void ParallelCoordinates::initializeGL(){
    curDraw = 0;
}

void ParallelCoordinates::resizeGL ( int width, int height ){
    curDraw = 0;
}

void ParallelCoordinates::mousePressEvent ( QMouseEvent * event ) {

    float selp = (float)event->pos().x() / (float)size().width() * 2.0f - 1.0f;

    int closest = 0;
    for(int i = 1; i < (int)dimLoc.size(); i++){
        if( fabsf( dimLoc[i].first-selp ) < fabsf( dimLoc[closest].first-selp ) ){
            closest = i;
        }
    }

    if( fabsf( dimLoc[closest].first-selp ) < 0.01f ){
        selected = closest;
        curDraw = 0;
    }

}

void ParallelCoordinates::mouseReleaseEvent ( QMouseEvent * event ) {

    if(selected != -1){
        float selp = (float)event->pos().x() / (float)size().width() * 2.0f - 1.0f;
        dimLoc[ selected ].first = selp;
        std::sort( dimLoc.begin(), dimLoc.end() );
        for(int i = 0; i < dim; i++){
            dimLoc[i].first = SCI::lerp( -1.0f, 1.0f, ((float)i+0.5f) / ((float)dim) );
        }
        selected = -1;
        curDraw = 0;
    }

}

void ParallelCoordinates::mouseMoveEvent ( QMouseEvent * event ) {

    if(selected != -1){
        float selp = (float)event->pos().x() / (float)size().width() * 2.0f - 1.0f;
        dimLoc[ selected ].first = selp;

        bool repeat;
        do {
            repeat = false;
            if( selected > 0 && dimLoc[ selected ].first < dimLoc[ selected-1 ].first ){
                std::swap( dimLoc[ selected ], dimLoc[ selected-1 ]);
                selected--;
                repeat = true;
            }
            if( selected < (dim-1) && dimLoc[ selected ].first > dimLoc[ selected+1 ].first ){
                std::swap( dimLoc[ selected ], dimLoc[ selected+1 ]);
                selected++;
                repeat = true;
            }
        } while(repeat);

        for(int i = 0; i < dim; i++){
            if( i != selected )
                dimLoc[i].first = SCI::lerp( -1.0f, 1.0f, ((float)i+0.5f) / ((float)dim) );
        }
        curDraw = 0;
    }
}

void ParallelCoordinates::paintGL(){

    glViewport( 0, 0, size().width(), size().height() );
    if( curDraw == 0 ){
        glClearColor( 0.95f,0.95f,0.95f,1 );
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    }

    if(pdata == 0 || color == 0){
        update();
        return;
    }

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( -1, 1, minv, maxv, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    if( curDraw == 0 ){
        glColor3f(1,1,1);
        glBegin(GL_QUADS);
            glVertex3f( -1,minv,0 );
            glVertex3f(  1,minv,0 );
            glVertex3f(  1,maxv,0 );
            glVertex3f( -1,maxv,0 );
        glEnd();


        glLineWidth(2.0f);
        glBegin(GL_LINES);
        for(int i = ((int)minv-1); i <= ((int)maxv+1); i++){
            glColor3f(0.85f,0.85f,0.85f);
            if((i%5)==0) glColor3f(0.55f,0.55f,0.55f);
            if(i==0) glColor3f(0.2f,0.2f,0.2f);
            glVertex3f( -1, i, 0 );
            glVertex3f(  1, i, 0 );
        }
        glEnd();

        glLineWidth(5.0f);
        glBegin(GL_LINES);
        for(int i = 0; i < (int)dimLoc.size(); i++){
            glColor3f(0,0,0);
            if(i == selected) glColor3f(1,0,0);
            glVertex3f( dimLoc[i].first, minv, 0 );
            glVertex3f( dimLoc[i].first, maxv, 0 );
        }
        glEnd();
    }


    glLineWidth(1.0f);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    SCI::Vex3 col;
    float * space = new float[dim];

    int step = pdata->GetElementCount() / 20000;

    if( curDraw == 0 ){
        for( ; curDraw < pdata->GetElementCount(); curDraw += step ){
            pdata->GetElement( curDraw, space );
            if( color != 0 ) col.SetFromUInt( color->GetColor( curDraw ) );

            glColor4f(col.x, col.y, col.z, 0.025f);
            glBegin(GL_LINE_STRIP);
            for(int j = 0; j < dim; j++){
                glVertex3f( dimLoc[j].first, space[ dimLoc[j].second ], 0 );
            }
            glEnd();
        }
        curDraw = 1;
    }
    else{
        int fin = curDraw+10000;
        for( ; curDraw < fin && curDraw < pdata->GetElementCount(); curDraw++ ){
            if( (curDraw%step) == 0 ) continue;
            pdata->GetElement( curDraw, space );
            if( color != 0 ) col.SetFromUInt( color->GetColor( curDraw ) );

            glColor4f(col.x, col.y, col.z, 0.025f);
            glBegin(GL_LINE_STRIP);
            for(int j = 0; j < dim; j++){
                glVertex3f( dimLoc[j].first, space[ dimLoc[j].second ], 0 );
            }
            glEnd();
        }
    }

    delete [] space;

    glDisable( GL_BLEND );

    update();

}

