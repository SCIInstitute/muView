#include <GL/glew.h>

#include <muView/ChartCloud.h>
#include <GL/oglCommon.h>
#include <QMouseEvent>

ChartCloud::ChartCloud(QObject *parent) : QGLWidget() {
    need_view_update = true;
    memset(clpX,0,sizeof(double)*4);
    memset(clpY,0,sizeof(double)*4);
    memset(clpZ,0,sizeof(double)*4);
    clpX[0] = clpY[1] = clpZ[2] = -1;
    useClipX = useClipY = useClipZ = false;
    proj.Set( 40.0f,40.0f, 0.1f,15.0f);
    mouse_active = false;
    parallel_coordinates = 0;


    mouse_button = Qt::NoButton;
    mouse_x = 0;
    mouse_y = 0;

    setMouseTracking(true);

    pln[0] = pln[1] = pln[2] = 0;


    /*

    QLineSeries *series = new QLineSeries();
    series->append(0, 6);
    series->append(2, 4);
    series->append(3, 8);
    series->append(7, 4);
    series->append(10, 5);
    *series << QPointF(11, 1) << QPointF(13, 3) << QPointF(17, 6) << QPointF(18, 3) << QPointF(20, 2);



    QChart *chart = new QChart();

    chart->legend()->hide();
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->setTitle("Simple line chart example");


    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);







    QImage * img = new QImage();
    *img = QImage(chartView->size().width(),chartView->size().height(), QImage::Format_ARGB32);


    QImage * imgSmall = new QImage();
    *imgSmall = img->scaledToHeight(height()/8.0);

    imgSmall->fill(QColor(230,230,230));

    QString path = "/Users/magdalenaschwarzl/Desktop/graph.png";
    imgSmall->save(path);

    allChartViews.push_back(chartView);
    */
    setAutoFillBackground(false);
}



void ChartCloud::initializeGL(){
    glewInit( );
}


void ChartCloud::createChartRects(int number)
{

    SCI::Vex3 location(1,2,3);
    std::vector<float> data;
    data.push_back(3);
    data.push_back(10);
    data.push_back(22);
    data.push_back(4);
    data.push_back(6);
    data.push_back(3);
    data.push_back(2);

    int numData = 7;


    for (int i = 0; i < number; ++i) {
        QPointF position(width()*(0.1 + (0.8*qrand()/(RAND_MAX+1.0))),
                        height()*(0.1 + (0.8*qrand()/(RAND_MAX+1.0))));

        chartRects.append(new ChartRect(position, location, &data[0], numData));
    }
}



void ChartCloud::paintEvent(QPaintEvent *event){

    makeCurrent();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();


    // Clear the display
    glClearColor(1,1,1,1);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    Draw( size().width(), size().height() );

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();






    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    foreach (ChartRect *chartRect, chartRects) {
        QImage * imgSmall = new QImage();
        * imgSmall = QImage(chartRect->width(),chartRect->height(), QImage::Format_ARGB32);
        *imgSmall = imgSmall->scaledToHeight(height()/8.0);

        imgSmall->fill(QColor(230,230,230));

        QPixmap pix = chartRect->grabChartView();
        int h = painter.window().height()/5;
        int w = painter.window().width()/5;
        chartRect->resize(w, h);


        chartRect->drawChartRect(&painter, pix, w, h);
    }
    painter.end();

}



void ChartCloud::showEvent(QShowEvent *event){

   Q_UNUSED(event);
   createChartRects(4);

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
        iso_tris->SortByPainters( iso_tris_ro, proj.GetMatrix() * pView->GetView() * tform, *iso_points );
    }
    if( draw_mode == 3 ){
        df_tris->SortByPainters( df_tris_ro, proj.GetMatrix() * pView->GetView() * tform, *df_points );
    }
    /*
    if(draw_mode == 4){
        edge_mesh->SortByPainters( edge_mesh_ro, proj.GetMatrix() * pView->GetView() * tform, *pmesh );
    }
    */
    tdata->tri_mesh.SortByPainters( tri_mesh_ro, proj.GetMatrix() * pView->GetView() * tform, *pmesh );
    for(int i = 0; i < addl_sil_mesh.size(); i++){
        addl_solid[i]->tri_mesh.SortByPainters( *(addl_mesh_ro[i]), proj.GetMatrix() * pView->GetView() * tform, *(addl_points[i]) );
    }


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

    proj.Set( proj.GetHFOV(), width, height, proj.GetHither(), proj.GetYon() );

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

    glViewport( 0,0,width,height );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMultMatrixf( proj.GetMatrix().data );

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
        //glPointSize(4.0f);
        //pmesh->Draw( SCI::Vex4(0,0,0,1) );
    glDisable(GL_DEPTH_TEST);
    }
    //////

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
      OVERVIEW WINDOW

    */
    float overviewWindow00 = width / 16;
    float overviewWindow01 = width / 8;
    float overviewWindow10 = height / 16;
    float overviewWindow11 = height / 8;



    // Draw border not aligned right now
    /*
    std::vector<SCI::Vex3> overviewWindow;

    overviewWindow.push_back(SCI::Vex3((overviewWindow00/width) -1,(overviewWindow10/height) -1,0));
    overviewWindow.push_back(SCI::Vex3((overviewWindow00/width) -1,(overviewWindow11/height) -1,0));
    overviewWindow.push_back(SCI::Vex3((overviewWindow01/width) -1,(overviewWindow10/height) -1,0));
    overviewWindow.push_back(SCI::Vex3((overviewWindow01/width) -1,(overviewWindow11/height) -1,0));


    glBegin(GL_LINE_LOOP);
    for(int i = 0; i < (int)overviewWindow.size(); i++){
        glColor3f(0,1,0);
        glVertex3fv( overviewWindow[i].data );
    }
    glEnd();
    */

    //draw Overview inside
    glViewport( overviewWindow00,overviewWindow10,overviewWindow01, overviewWindow11);
    glScissor( overviewWindow00,overviewWindow10,overviewWindow01, overviewWindow11);
    glClearColor(0,0,1,0.05);
    glEnable(GL_SCISSOR_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

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
    pView->Save("view.txt");
    mouse_active = ( state == QMouseEvent::MouseButtonPress );
    return mouse_active;
}


bool ChartCloud::MouseMotion(int button, int x, int dx, int y, int dy){
    if(mouse_active){
        if(button == Qt::LeftButton)   pView->Rotate(-(float)(dx),(float)(dy));
        if(button == Qt::MiddleButton) pView->Translate((float)(dx),(float)(dy));
        if(button == Qt::RightButton)  pView->Zoom((float)(dy));
        need_view_update = true;
        update();
    }
    return mouse_active;
}

bool ChartCloud::Keypress( int key, int x, int y ){
    return false;
}


void ChartCloud::Initialize( ){ }

