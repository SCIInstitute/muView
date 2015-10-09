#include <Drawable/DimRedDisplay.h>
#include <SCI/Vex3.h>
#include <QColorDialog>

#include <GL/oglCommon.h>

void DrawCircle( SCI::Vex2 pos, float rad, int type ){
    glBegin(type);
    for(int i = 0; i <= 20; i++){
        float t = (float)(i%20)/20.0f;
        float ang = 2.0f*3.14159265f*t;
        float x = pos.x + cosf(ang)*rad;
        float y = pos.y + sinf(ang)*rad;
        glVertex3f(x,y,0);
    }
    glEnd();
}

using namespace Drawable;

DimRedDisplay::DimRedDisplay(){
    prov    = 0;
    mouse_active = false;
    sel_min = SCI::VEX2_INVALID;
    sel_max = SCI::VEX2_INVALID;
    mp      = SCI::VEX2_INVALID;
    mouse_x = mouse_y = 0;
    mouse_hold_timer.Start();
    line_active = false;
}

void DimRedDisplay::Initialize( Provenance & _prov, Data::BasicData & _data ){
    prov = &_prov;
    draw_bar.SetMinimumValue(_data.GetMinimumValue());
    draw_bar.SetMaximumValue(_data.GetMaximumValue());
}

void DimRedDisplay::Draw(){

    SetViewport();

    Clear();

    // Draw Prov Data
    if(prov){
        glColor3f(1,1,1);
        prov->GetCurrentNode()->GetDrawingSurface().GetColorTexture(0)->DrawTexture();

        glPointSize(16.0f);
        glColor3f(0,0,0);
        prov->GetCurrentNode()->GetLine().DrawSelected();
        glPointSize(12.0f);
        glLineWidth(3.0f);
        glColor3f(0,0,1);
        prov->GetCurrentNode()->GetLine().Draw();
    }

    // Draw circle around histogram area
    if(mp.isValid()){
        glLineWidth(1.0f);
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
            glColor4f( 0.8f,0,0,0.5f);
            DrawCircle( mp, 0.05f, GL_POLYGON );
        glDisable(GL_BLEND);
        glColor4f( 0.8f,0,0,1.0f);
        DrawCircle( mp, 0.05f, GL_LINE_STRIP );
    }

    // Selecting a region
    if(sel_min.isValid()){
        glLineWidth(2.0f);
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
        glColor4f(0.2f,0.2f,0.2f,0.5f);
        glBegin(GL_QUADS);
            glVertex3f( sel_min.x, sel_min.y, 0 );
            glVertex3f( sel_max.x, sel_min.y, 0 );
            glVertex3f( sel_max.x, sel_max.y, 0 );
            glVertex3f( sel_min.x, sel_max.y, 0 );
        glEnd();
        glDisable(GL_BLEND);

        glColor3f(0,0,1);
        glBegin(GL_LINE_LOOP);
            glVertex3f( sel_min.x, sel_min.y, 0 );
            glVertex3f( sel_max.x, sel_min.y, 0 );
            glVertex3f( sel_max.x, sel_max.y, 0 );
            glVertex3f( sel_min.x, sel_max.y, 0 );
        glEnd();

    }

    // Draw frame around viewport
    DrawFrame();

    if( mouse_hold_timer.Tick() > 1.0f && isPointInFrame( mouse_x,mouse_y ) ){

        SCI::Vex2 pos = SCI::Vex2(mouse_x-u0,mouse_y-v0) / SCI::Vex2(width,height) * 2.0f - 1.0f;
        int ecnt = 0;
        if( pos.x > sel_min.x && pos.x < sel_max.x && pos.y > sel_min.y && pos.y < sel_max.y ){
            ecnt = UpdateAverageInBBox( );
        }
        else{
            ecnt = UpdateAverageInRadius( );
        }

        if( ecnt == 0 ){
            selected_voxel_id = -1;
            mp = SCI::VEX2_INVALID;
        }
        else{
            selected_voxel_id = 0;
        }

        int bar_w = 200;
        int bar_h = 100;
        int bar_u0 = mouse_x-200-10;
        int bar_v0 = mouse_y+10;

        if(selected_voxel_id >= 0){
            draw_bar.SetShape( bar_u0, bar_v0, bar_w, bar_h );
            draw_bar.Draw();
        }
    }


}

int DimRedDisplay::UpdateAverageInRadius( ){
    mp = SCI::Vex2(mouse_x-u0,mouse_y-v0) / SCI::Vex2(width,height) * 2.0f - 1.0f;
    std::vector<int> plist = prov->GetCurrentNode()->GetHistogram().GetBinList(mp.x,mp.y,0.05f);
    draw_bar.SetAverage( prov->GetCurrentNode()->GetHistogram().GetBinAvg( plist ) );
    draw_bar.SetMinimum( prov->GetCurrentNode()->GetHistogram().GetBinMin( plist ) );
    draw_bar.SetMaximum( prov->GetCurrentNode()->GetHistogram().GetBinMax( plist ) );
    return prov->GetCurrentNode()->GetHistogram().GetBinCnt( plist );
}


int DimRedDisplay::UpdateAverageInBBox( ){
    std::vector<int> plist = prov->GetCurrentNode()->GetHistogram().GetBinList(sel_min.x,sel_min.y,sel_max.x,sel_max.y);
    draw_bar.SetAverage( prov->GetCurrentNode()->GetHistogram().GetBinAvg( plist ) );
    draw_bar.SetMinimum( prov->GetCurrentNode()->GetHistogram().GetBinMin( plist ) );
    draw_bar.SetMaximum( prov->GetCurrentNode()->GetHistogram().GetBinMax( plist ) );
    return prov->GetCurrentNode()->GetHistogram().GetBinCnt( plist );
}


bool DimRedDisplay::MouseClick(int button, int state, int x, int y){
    SCI::Vex2 pos = SCI::Vex2(x-u0,y-v0) / SCI::Vex2(width,height) * 2.0f - 1.0f;
    mouse_active = false;

    if(!isPointInFrame(x,y)){ return false; }

    if( pos.x < -1 || pos.x > 1 || pos.y < -1 || pos.y > 1 ) return false;

    if(button == LeftButton && state == Down){
        if( pos.x > sel_min.x && pos.x < sel_max.x && pos.y > sel_min.y && pos.y < sel_max.y ){
            // do nothing
        }
        else if( x >= u0 && x < (u0+width) && y >= v0 && y < (v0+height) ){
            mouse_active = true;
            sel_min = sel_max = pos;
        }
    }
    if(button == LeftButton && state == Double){
        BBox bbox;
        bbox.bbmin = (prov->GetCurrentNode()->GetTransform().Inverse()*SCI::Vex3(sel_min,0)).xy();
        bbox.bbmax = (prov->GetCurrentNode()->GetTransform().Inverse()*SCI::Vex3(sel_max,0)).xy();
        prov->Push( bbox );
        sel_min = sel_max = SCI::VEX2_INVALID;
        return true;
    }
    if(button == LeftButton && state == Up){
        SCI::Vex2 real_min = SCI::Vex2( STD_MIN(sel_min.x,sel_max.x), STD_MIN(sel_min.y,sel_max.y) );
        SCI::Vex2 real_max = SCI::Vex2( STD_MAX(sel_min.x,sel_max.x), STD_MAX(sel_min.y,sel_max.y) );
        sel_min = real_min;
        sel_max = real_max;
    }

    line_active = false;
    if(button == RightButton && state == Down){
        SCI::Vex2 p = SCI::Vex2(x-u0,y-v0) / SCI::Vex2(width,height) * 2.0f - 1.0f;
        if( prov->GetCurrentNode()->GetLine().DistanceToHead(p) < 0.05f || prov->GetCurrentNode()->GetLine().DistanceToTail(p) < 0.05f ){
            prov->GetCurrentNode()->GetLine().SelectClosestControlPoint( p );
        }
        else{
            prov->GetCurrentNode()->GetLine().Clear();
            line_active = true;
            return true;
        }
    }
    if(button == RightButton && state == Double){
        SCI::Vex2 p = SCI::Vex2(x-u0,y-v0) / SCI::Vex2(width,height) * 2.0f - 1.0f;
        if( prov->GetCurrentNode()->GetLine().DistanceToHead(p) < 0.05f || prov->GetCurrentNode()->GetLine().DistanceToTail(p) < 0.05f ){
            prov->GetCurrentNode()->GetLine().SelectClosestControlPoint( p );
            prov->GetCurrentNode()->GetLine().SelectedControlPointDialog();
            return true;
        }
    }
    if(button == RightButton && state == Up){
        line_active = false;
        return true;
    }
    return mouse_active;
}

bool DimRedDisplay::MouseMotion(int button, int x, int dx, int y, int dy){
    if( mouse_active ){
        x = STD_MIN( STD_MAX( x-u0, 0 ), width -1 );
        y = STD_MIN( STD_MAX( y-v0, 0 ), height-1 );
        sel_max = SCI::Vex2(x,y) / SCI::Vex2(width,height) * 2.0f - 1.0f;
    }

    if(mouse_x!=x || mouse_y!=y){
        selected_voxel_id = -2;
        mouse_hold_timer.Start();
        mouse_x = x;
        mouse_y = y;
        mp = SCI::VEX2_INVALID;
    }

    if(line_active){
        prov->GetCurrentNode()->GetLine().AddPoint( SCI::Vex2(x-u0,y-v0) / SCI::Vex2(width,height) * 2.0f - 1.0f );
        return true;
    }


    return mouse_active;
}


