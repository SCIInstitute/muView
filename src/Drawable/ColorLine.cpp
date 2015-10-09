#include <Drawable/ColorLine.h>

#include <GL/oglCommon.h>
#include <QColorDialog>

inline std::pair<float,float> PointSegmentDistance(SCI::Vex2 v0, SCI::Vex2 v1, SCI::Vex2 p){
    SCI::Vex2 v1v0 = v1-v0;
    SCI::Vex2 v0p  = v0-p;
    SCI::Vex2 v1p  = v1-p;

    float u = -SCI::dot(v0p,v1v0) / SCI::dot(v1v0,v1v0);

    if(u >= 1){ return std::pair<float,float>( 1, v1p.Length() ); }
    if(u <= 0){ return std::pair<float,float>( 0, v0p.Length() ); }

    SCI::Vex2 p_close = SCI::lerp(v0,v1,u);

    return std::pair<float,float>( u, (p_close - p).Length() );
}



using namespace Drawable;

ColorLine::ColorLine(){
    selected = -1;
}

void ColorLine::Clear(){
    input.clear();
    simplified.clear();
    selected = -1;
}

void ColorLine::AddPoint( SCI::Vex2 p ){
    input.push_back( SCI::Vex3( p, 0 ) );
    simplified.Calculate( input, 0.05f, 1024 );
    colors = std::vector< SCI::Vex4 >( simplified.size(), SCI::Vex4(1,0,1,0.1f) );
    total_length = 0;
    for(int i = 1; i < (int)simplified.size(); i++){
        total_length += (simplified[i]-simplified[i-1]).Length();
    }
}

float ColorLine::GetParameterization( SCI::Vex2 p ){
    float dist = FLT_MAX;
    float param = FLT_MAX;
    float curr_length = 0;
    for(int i = 1; i < (int)simplified.size(); i++){
        std::pair<float,float> df = PointSegmentDistance( simplified[i-1].xy(), simplified[i].xy(), p );
        float t = df.first;
        float d = df.second;
        float l = (simplified[i-1].xy()-simplified[i].xy()).Length();
        if( d < dist ){
            dist = d;
            param = (curr_length+l*t)/total_length;
        }
        curr_length += l;
    }
    return param;
}

float ColorLine::DistanceToHead( SCI::Vex2 p ){
    if( simplified.size() == 0 ){ return FLT_MAX; }
    return (simplified[0].xy()-p).Length();
}

float ColorLine::DistanceToTail( SCI::Vex2 p ){
    if( simplified.size() == 0 ){ return FLT_MAX; }
    return (simplified.back().xy()-p).Length();
}

void ColorLine::SelectClosestControlPoint( SCI::Vex2 p ){
    if( simplified.size() == 0 ){
        selected = -1;
    }
    else{
        if( DistanceToHead(p) < DistanceToTail(p) ){
            selected = 0;
        }
        else{
            selected = simplified.size()-1;
        }
    }
}

void ColorLine::Draw( ){
    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < (int)simplified.size(); i++){
        glVertex3f( simplified[i].x, simplified[i].y, 0 );
    }
    glEnd();
    if(simplified.size()>0){
        glBegin(GL_POINTS);
            glVertex3f( simplified[0].x, simplified[0].y, 0 );
            glVertex3f( simplified.back().x, simplified.back().y, 0 );
        glEnd();
    }
}

void ColorLine::DrawSelected( ){
    if(selected>=0){
        glBegin(GL_POINTS);
            glVertex3f( simplified[selected].x, simplified[selected].y, 0 );
        glEnd();
    }
}

void ColorLine::SelectedControlPointDialog( ){
    if(selected>=0){
        QColor col = QColor((int)(colors[selected].x*255.0f),(int)(colors[selected].y*255.0f),(int)(colors[selected].z*255.0f),(int)(colors[selected].w*255.0f));
        col = QColorDialog::getColor(col, 0, "Control Point Color",  QColorDialog::ShowAlphaChannel | QColorDialog::DontUseNativeDialog);
        if(col.isValid()){
            colors[selected] = SCI::Vex4(col.red(),col.green(),col.blue(),col.alpha()) / 255.0f;
            colors[selected].Print(); fflush(stdout);
        }
    }
}

SCI::Vex4 ColorLine::GetColor( SCI::Vex2 p ){
    if(simplified.size()==0) return SCI::Vex4(1,0,1,0.1f);
    return lerp( colors[0], colors.back(), GetParameterization( p ) );
}

