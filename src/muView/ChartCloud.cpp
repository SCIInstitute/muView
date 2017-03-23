#include <GL/glew.h>
//#include <GLFW/glfw3.h>

#include <muView/ChartCloud.h>
#include <GL/oglCommon.h>
#include <QScreen>
#include <QGuiApplication>

ChartCloud::ChartCloud(QObject *parent) : QGLWidget() {


    need_view_update = true;
    memset(clpX,0,sizeof(double)*4);
    memset(clpY,0,sizeof(double)*4);
    memset(clpZ,0,sizeof(double)*4);
    clpX[0] = clpY[1] = clpZ[2] = -1;
    useClipX = useClipY = useClipZ = false;
    //projPersp.Set( 40.0f,40.0f, 0.1f,15.0f);
    mouse_active = false;
    parallel_coordinates = 0;

    left = -1.1f;
    right = 1.1f;
    bottom = -1.1f;
    top = 1.1f;
    near = 0.1f;
    far = 15.0f;
    projOrtho.Set(left, right, bottom, top, near, far);
    projOrthoPlain.Set(left, right, bottom, top, near, far);

    factor = ((QGuiApplication*)QCoreApplication::instance())
            ->primaryScreen()->devicePixelRatio();

    mouse_button = Qt::NoButton;
    mouse_x = 0;
    mouse_y = 0;

    setMouseTracking(true);

    pln[0] = pln[1] = pln[2] = 0;


    pView = new SCI::ThirdPersonCameraControls();
    pViewRotationOnly = new SCI::ThirdPersonCameraControls();
    rawProj = new SCI::ThirdPersonCameraControls();

    /*
    pView->Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    pViewRotationOnly->Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    rawProj->Set( 15.0f, 75.0f, 5.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    */


    pView->Set( 0.0f, 0.0f, 3.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    pViewRotationOnly->Set( 0.0f, 0.0f, 3.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );
    rawProj->Set( 0.0f, 0.0f,3.0f, SCI::Vex3(0,0,0), SCI::Vex3(0,1,0) );


    zoomFactor=1;
    translationFactorX=0;
    translationFactorY=0;

    chartRatioWidth = 1/5.0;
    chartRatioHeight = 1/5.0;

    setAutoFillBackground(false);
}



void ChartCloud::initializeGL(){
    glewInit( );
    //glfwInit();


    /* OPENGL VERISON
    int major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    std::cout << "OPENGL version " << major << "." << minor << std::endl;
    */
}


void ChartCloud::mapToScreen(float& x, float& y, SCI::Vex3 location){

    SCI::Vex4 screen = projOrtho.GetMatrix()*pView->GetView()*tform*SCI::Vex4(location.x, location.y, location.z, 1);
    screen/=screen.w;

    x = screen.x;
    y = screen.z;

    //std::cout << "x,y " << x << ", " << y << std::endl;

}

void ChartCloud::redoChartRects(){

    int dataIndex;
    float x,y=-2;

    int w = width();
    int h = height();


    std::vector<float> xList;
    std::vector<float> yList;

    SCI::Vex3 location;
    std::vector<float> chartData;
    int numData = 0;
    bool onScreen;

     for (int i = 0; i < chartRects.size(); ++i) {

         onScreen = false;

         while(!onScreen){
             dataIndex = rand() % pdata->GetElementCount();
             location = pmesh->GetVertex(dataIndex);
             mapToScreen(x,y,location);
             //std::cout<<"zoom: " << zoomFactor << "  iterating " << itCount << "... x,y" << x << ", " << y << std::endl;

             //find new point if not on screen
             onScreen = !(x < -1 || x > 0.9 || y < -1 || y > 0.65);
             bool overlapX = false;
             bool overlapY = false;

             //check overlap with other charts
             if(onScreen){
                 for (int iComp=0; iComp< xList.size(); iComp++){
                    if(fabs(x - xList[iComp]) < (chartRatioWidth*2.0)) {
                        //onScreen  = false;
                        std::cout << "overlap x "<< x << " - " << xList[iComp] << "diff: " << chartRatioWidth << std::endl;
                        overlapX = true;
                    }
                    if(fabs(y - yList[iComp]) < (chartRatioHeight*2.0)){
                        //onScreen  = false;
                        overlapY = true;
                        std::cout << "overlap y " << y << " - " << yList[iComp] << "diff: " << chartRatioHeight << std::endl;
                    }
                    if(overlapX && overlapY){
                        onScreen  = false;
                        std::cout << "both overlkap" << std::endl;
                    }
                 }
             }
         }






         if(xList.size() > i){
             xList[i] = x;
             yList[i] = y;
         }
         else{
             xList.push_back(x);
             yList.push_back(y);
         }


         pdata->GetData(dataIndex,chartData);
         numData = chartData.size();

         //std::cout << "data index: " << dataIndex << std::endl;


         x = (x+1)*0.5;
         y = (y+1)*0.5;

         QPointF position(x*width(),y*height());
         //std::cout<< "found  x,y " << x << ", " << y << std::endl;

         chartRects[i]->setPosition(position);
         chartRects[i]->setData(&chartData[0], chartData.size());
     }

 }


void ChartCloud::createChartRects(int number)
{

    float steps = 30;
    int w = width();
    int h = height();
    float x,y=0;
    float globalMIN = pdata->GetGlobalMin();
    float globalMAX = pdata->GetGlobalMax();
    int dataIndex=0;
    std::vector<float> chartData;
    SCI::Vex3 location;
    int numData;

    for (int i = 0; i < number; ++i) {

        dataIndex = rand() % pdata->GetElementCount();

        dataIndex = i;

        pdata->GetData(dataIndex,chartData);
        numData = chartData.size();


        location = pmesh->GetVertex(dataIndex);
        mapToScreen(x,y,location);

        x = (x+1)*0.5;
        y = (y+1)*0.5;

        QPointF position(x*width(),y*height());
        chartRects.append(new ChartRect(position,  &chartData[0], numData, chartRatioWidth*w, chartRatioHeight*h, globalMIN, globalMAX, steps));
    }
}



void ChartCloud::paintEvent(QPaintEvent *event){

    makeCurrent();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();


    // Clear the display
    glClearColor(1,1,1,1);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );


    Draw(width()*factor, height() *factor);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();



    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    foreach (ChartRect *chartRect, chartRects) {
        QImage * imgSmall = new QImage();
        * imgSmall = QImage(chartRect->width(),chartRect->height(), QImage::Format_ARGB32);
        *imgSmall = imgSmall->scaledToHeight(height()/8.0);

        imgSmall->fill(QColor(230,230,230, 125));

        QPixmap pix = chartRect->grabChartView();
        chartRect->drawChartRect(&painter, pix);
    }
    painter.end();

}



void ChartCloud::showEvent(QShowEvent *event){

   Q_UNUSED(event);
   createChartRects(2);

}


void ChartCloud::mouseDoubleClickEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonDblClick, x, y );

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void ChartCloud::mouseMoveEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseMotion( mouse_button, x, x-mouse_x, y, y-mouse_y );

    mouse_x = x;
    mouse_y = y;

}

void ChartCloud::mousePressEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonPress, x, y );

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void ChartCloud::mouseReleaseEvent ( QMouseEvent * event ){

    int x = event->x();
    int y = size().height()-event->y()-1;

    MouseClick( event->button(), QMouseEvent::MouseButtonRelease, x, y );

    mouse_button = Qt::NoButton;
    mouse_x = x;
    mouse_y = y;
}

void ChartCloud::keyPressEvent ( QKeyEvent * event ){

    Keypress( event->key(), mouse_x, mouse_y );

}

void ChartCloud::keyReleaseEvent ( QKeyEvent * event ){

}

void ChartCloud::UpdateView( ){
    float scl = 2.0f / pmesh->bb.GetMaximumDimensionSize();
    tform.LoadIdentity();
    tform *= SCI::Mat4( SCI::Mat4::MAT4_SCALE, scl, scl, scl );
    tform *= SCI::Mat4( SCI::Mat4::MAT4_TRANSLATE, -pmesh->bb.GetCenter() );

    if( draw_mode == 2 ){
        iso_tris->SortByPainters( iso_tris_ro, projOrtho.GetMatrix() * pView->GetView() * tform, *iso_points );
    }
    if( draw_mode == 3 ){
        df_tris->SortByPainters( df_tris_ro, projOrtho.GetMatrix() * pView->GetView() * tform, *df_points );
    }
    /*
    if(draw_mode == 4){
        edge_mesh->SortByPainters( edge_mesh_ro, projOrtho.GetMatrix() * pView->GetView() * tform, *pmesh );
    }
    */
    tdata->tri_mesh.SortByPainters( tri_mesh_ro, projOrtho.GetMatrix() * pView->GetView() * tform, *pmesh );
    for(int i = 0; i < addl_sil_mesh.size(); i++){
        addl_solid[i]->tri_mesh.SortByPainters( *(addl_mesh_ro[i]), projOrtho.GetMatrix() * pView->GetView() * tform, *(addl_points[i]) );
    }

    redoChartRects();

    need_view_update = false;
}

void ChartCloud::resizeGL(int width, int height){
    setupViewport(width, height);
}

void ChartCloud::setupViewport(int width, int height){
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
#ifdef QT_OPENGL_ES
    glOrthof(-0.5, +0.5, -0.5, 0.5, 4.0, 15.0);
#else
    glOrtho(-0.5, +0.5, -0.5, 0.5, 4.0, 15.0);
#endif
    glMatrixMode(GL_MODELVIEW);
}

void ChartCloud::Draw( int width, int height ){

    if(pmesh == 0 || tdata == 0) return;




    //TODO
    //proj.Set( proj.GetHFOV(), width, height, proj.GetHither(), proj.GetYon() );

    if( need_view_update ){
        UpdateView();
    }

    if( cluster->isRunning() ){
        colormap->resize(pdata->GetVoxelCount());
        for(int i = 0; i < pdata->GetVoxelCount(); i++){
            colormap->at(i) = cat_cmap->GetColor( cluster->GetClusterID(i) ).UIntColor();
        }
        parallel_coordinates->Reset();

    }

    //todo
    glViewport( 0,0,width, height);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMultMatrixf( projOrtho.GetMatrix().data );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glMultMatrixf( pView->GetView().data );

    if( useClipX){ glEnable( GL_CLIP_PLANE0 ); glClipPlane( GL_CLIP_PLANE0, clpX ); }
    if( useClipY){ glEnable( GL_CLIP_PLANE1 ); glClipPlane( GL_CLIP_PLANE1, clpY ); }
    if( useClipZ){ glEnable( GL_CLIP_PLANE2 ); glClipPlane( GL_CLIP_PLANE2, clpZ ); }

    glMultMatrixf( tform.data );


    glLineWidth(1.0f);


    // draw_mode = 0
    // This draws the colored volum
    {
    glEnable(GL_DEPTH_TEST);
        glPointSize(3.0f);
        if( pdata == 0 ){
            pmesh->Draw( SCI::Vex4(1.0f, 0.95f, 1.0f, 1.0f ) );
        }
        else{
            pmesh->Draw( *colormap );
        }
    glDisable(GL_DEPTH_TEST);
    }

    // This draws the black contours
    glDisable(GL_DEPTH_TEST);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );
        RecalculateSil();
        tdata->tri_mesh.Draw( tri_mesh_ro, *pmesh, sil_color );
        for(int i = 0; i < addl_sil_mesh.size(); i++){
            addl_solid[i]->tri_mesh.Draw( *(addl_mesh_ro[i]), *(addl_points[i]), *(addl_sil_color[i]) );
        }
    glDisable( GL_BLEND );









    /*
      OVERVIEW WINDOW starts here

    float overviewWindow00 = width / 6;
    float overviewWindow01 = width / 3;
    float overviewWindow10 = height / 6;
    float overviewWindow11 = height / 3;

    */



    float overviewWindow00 = width / 16;
    float overviewWindow01 = width / 8;
    float overviewWindow10 = height / 16;
    float overviewWindow11 = height / 8;


    //find current near plane
    SCI::Vex4 left_up =     tform.Inverse() * pView->GetView().Inverse() * projOrtho.GetMatrix().Inverse() * SCI::Vex4(-1,-1,0.9,1);
    SCI::Vex4 left_down =   tform.Inverse() * pView->GetView().Inverse() * projOrtho.GetMatrix().Inverse() * SCI::Vex4(-1, 1,0.9,1);
    SCI::Vex4 right_up =    tform.Inverse() * pView->GetView().Inverse() * projOrtho.GetMatrix().Inverse() * SCI::Vex4( 1,-1,0.9,1);
    SCI::Vex4 right_down =  tform.Inverse() * pView->GetView().Inverse() * projOrtho.GetMatrix().Inverse() * SCI::Vex4( 1, 1,0.9,1);


    left_up = left_up/left_up.w;
    right_down = right_down/right_down.w;
    left_down = left_down/left_down.w;
    right_up = right_up/right_up.w;

    /*
    THIS Should be the near plane quad
    */
    std::vector<SCI::Vex3> quad1;
    quad1.push_back(SCI::Vex3(left_up.x,left_up.y,left_up.z));
    quad1.push_back(SCI::Vex3(left_down.x,left_down.y,left_down.z));
    quad1.push_back(SCI::Vex3(right_down.x,right_down.y,right_down.z));
    quad1.push_back(SCI::Vex3(right_up.x,right_up.y,right_up.z));

    std::vector<SCI::Vex3> colors;
    colors.push_back(SCI::Vex3(0,0,0));
    colors.push_back(SCI::Vex3(1,0,0));
    colors.push_back(SCI::Vex3(0,1,0));
    colors.push_back(SCI::Vex3(0,0,1));



/*
    // Draw it in the big window
    // shuold fill the whole screen
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUADS);
    for(int i = 0; i < (int)quad1.size(); i++){
        glColor3f(colors[i].data[0],colors[i].data[1],colors[i].data[2]);
        glVertex3fv( quad1[i].data );
    }
    glEnd();
*/



















    //draw Overview inside
    glViewport( overviewWindow00,overviewWindow10,overviewWindow01, overviewWindow11);
    glScissor( overviewWindow00,overviewWindow10,overviewWindow01, overviewWindow11);
    glClearColor(0.2,0.2,0.2,0.05);
    glEnable(GL_SCISSOR_TEST);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );


    // set projection matrix to orthographic projection

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMultMatrixf( projOrthoPlain.GetMatrix().data );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glMultMatrixf( pViewRotationOnly->GetView().data );
    glMultMatrixf( tform.data );

    glDisable(GL_DEPTH_TEST);
    {
        glPointSize(3.0f);
        if( pdata == 0 ){
            pmesh->Draw( SCI::Vex4(1.0f, 0.95f, 1.0f, 1.0f ) );
        }
        else{
            pmesh->Draw( *colormap );
        }
    }


    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glMultMatrixf( pViewRotationOnly->GetView().data );
    glMultMatrixf( tform.data );


    /*
    // Draw the near plane in the overview window
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    for(int i = 0; i < (int)quad1.size(); i++){
        glColor4f(0,1,0,0.5);
        glVertex3fv( quad1[i].data );
    }
    glEnd();
    */




    SCI::Vex4 left_up_overview =     tform.Inverse() * pViewRotationOnly->GetView().Inverse() * projOrthoPlain.GetMatrix().Inverse() * SCI::Vex4(-1,-1,0.9,1);
    SCI::Vex4 left_down_overview  =   tform.Inverse() * pViewRotationOnly->GetView().Inverse() * projOrthoPlain.GetMatrix().Inverse() * SCI::Vex4(-1, 1,0.9,1);
    SCI::Vex4 right_up_overview  =    tform.Inverse() * pViewRotationOnly->GetView().Inverse() * projOrthoPlain.GetMatrix().Inverse() * SCI::Vex4( 1,-1,0.9,1);
    SCI::Vex4 right_down_overview  =  tform.Inverse() * pViewRotationOnly->GetView().Inverse() * projOrthoPlain.GetMatrix().Inverse() * SCI::Vex4( 1, 1,0.9,1);


    left_up_overview  = left_up_overview /left_up_overview.w;
    right_down_overview  = right_down_overview /right_down_overview.w;
    left_down_overview  = left_down_overview /left_down_overview.w;
    right_up_overview  = right_up_overview /right_up_overview.w;

    /*
    THIS Should be the near plane quad
    */
    std::vector<SCI::Vex3> quads_overview ;
    quads_overview.push_back(SCI::Vex3(left_down.x,left_down.y,left_down.z));
    quads_overview.push_back(SCI::Vex3(left_down_overview.x,left_down_overview.y,left_down_overview.z));
    quads_overview.push_back(SCI::Vex3(right_down_overview.x,right_down_overview.y,right_down_overview.z));
    quads_overview.push_back(SCI::Vex3(right_down.x,right_down.y,right_down.z));


    quads_overview.push_back(SCI::Vex3(left_down.x,left_down.y,left_down.z));
    quads_overview.push_back(SCI::Vex3(left_down_overview.x,left_down_overview.y,left_down_overview.z));
    quads_overview.push_back(SCI::Vex3(left_up_overview.x,left_up_overview.y,left_up_overview.z));
    quads_overview.push_back(SCI::Vex3(left_up.x,left_up.y,left_up.z));


    quads_overview.push_back(SCI::Vex3(right_up.x,right_up.y,right_up.z));
    quads_overview.push_back(SCI::Vex3(right_up_overview.x,right_up_overview.y,right_up_overview.z));
    quads_overview.push_back(SCI::Vex3(left_up_overview.x,left_up_overview.y,left_up_overview.z));
    quads_overview.push_back(SCI::Vex3(left_up.x,left_up.y,left_up.z));


    quads_overview.push_back(SCI::Vex3(right_up.x,right_up.y,right_up.z));
    quads_overview.push_back(SCI::Vex3(right_up_overview.x,right_up_overview.y,right_up_overview.z));
    quads_overview.push_back(SCI::Vex3(right_down_overview.x,right_down_overview.y,right_down_overview.z));
    quads_overview.push_back(SCI::Vex3(right_down.x,right_down.y,right_down.z));



    // Focus the near plane in the overview window
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    for(int i = 0; i < (int)quads_overview.size(); i++){
        glColor4f(1,1,1,0.7);
        glVertex3fv(quads_overview[i].data );
    }
    glEnd();


    glDisable(GL_SCISSOR_TEST);
    glClearColor(1,1,1,1);









}


void ChartCloud::RecalculateSil(){
    if( tdata != 0 && pmesh != 0 ){
        if( sil_mesh.size() == 0 ){
            sil_mesh.Calculate( tdata->tri_mesh, *pmesh );
            //sil_mesh.Smooth( tdata->tri_mesh, 1 );
            sil_mesh.ZeroFlatRegions(tdata->tri_mesh, *pmesh );
        }
        sil_color.SetFromOrientation( sil_mesh, pView->GetVD(), 7.5f );
    }

    for(int i = 0; i < (int)addl_points.size() && i < (int)addl_solid.size(); i++){
        if( (int)addl_sil_mesh.size()  <= i ){ addl_sil_mesh.push_back( new Data::Mesh::OrientationMesh() ); }
        if( (int)addl_sil_color.size() <= i ){ addl_sil_color.push_back( new Data::Mesh::ColorMesh() ); }
        if( (int)addl_mesh_ro.size()   <= i ){ addl_mesh_ro.push_back( new std::vector<int>() ); }
        if( (int)addl_sil_mesh[i]->size() == 0 ){
            addl_sil_mesh[i]->Calculate( addl_solid[i]->tri_mesh, *(addl_points[i]) );
            //addl_sil_mesh[i]->Smooth( tdata->tri_mesh, 1 );
            addl_sil_mesh[i]->ZeroFlatRegions(addl_solid[i]->tri_mesh, *(addl_points[i]) );
        }
        addl_sil_color[i]->SetFromOrientation( *(addl_sil_mesh[i]), pView->GetVD(), 2.0f );
    }
}

bool ChartCloud::MouseClick(int button, int state, int x, int y){
    //pView->Save("view.txt");
    mouse_active = ( state == QMouseEvent::MouseButtonPress );

    return mouse_active;
}




bool ChartCloud::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if((button == Qt::LeftButton && QApplication::keyboardModifiers() && Qt::ControlModifier) || button == Qt::MiddleButton){
            pView->Translate((float)(dx),(float)(dy));
            translationFactorX = translationFactorX + dx;
            translationFactorY = translationFactorY + dy;
            //std::cout << "translate " << std::endl;
        }else{
            if(button == Qt::LeftButton) {
                pView->Rotate(-(float)(dx),(float)(dy));
                pViewRotationOnly->Rotate(-(float)(dx),(float)(dy));
                //std::cout << "rotate " << std::endl;
            }
            if(button == Qt::RightButton){

                zoomFactor = zoomFactor + dy*0.001;
                zoomFactor = fmax(0.2,zoomFactor);
                zoomFactor = fmin(10,zoomFactor);
                projOrtho.Set(left*zoomFactor, right*zoomFactor, bottom*zoomFactor, top*zoomFactor, near*zoomFactor, far*zoomFactor);

             }
        }
        need_view_update = true;
        update();
    }
    return mouse_active;
}

bool ChartCloud::Keypress( int key, int x, int y ){
    return false;
}


void ChartCloud::Initialize( ){ }

