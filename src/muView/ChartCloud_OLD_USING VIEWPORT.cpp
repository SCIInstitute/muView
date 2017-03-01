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


    allCharts.push_back(chartView);



    QString path2 = "/Users/magdalenaschwarzl/Desktop/testImage.png";
    QImage * img = new QImage();
    if(! img->load(path2)){
        std::cout << "creating image..." << std::endl;
        *img = QImage(chartView->size().width(),chartView->size().height(), QImage::Format_ARGB32);
    }


    img->fill(QColor(200,70,120));
    QPainter painter(img);

    QPixmap pix = chartView->grab();
    int h = painter.window().height();
    int w = painter.window().width();
    chartView->resize(w, h);
    painter.drawPixmap(0,0, w, h, pix);

    if(!img->save(path2)){
        std::cout << "ERROR saving image..." << std::endl;
    }

   textures.push_back(0);
    myLoadTexture(path2, textures.back());
}


void ChartCloud::myLoadTexture(QString filename, int texID){


    QImage t;
    QImage b;

    if ( !b.load( filename ) )
    {
      b.load( "/Users/magdalenaschwarzl/Desktop/error.png" );
      //b = QImage( 20,20, QImage::Format_ARGB32 );
      //b.fill( Qt::green.rgb() );

    }
b.save("/Users/magdalenaschwarzl/Desktop/usedTexture.png");
    t = QGLWidget::convertToGLFormat( b );
    glGenTextures( 1, &textures[texID] );
    glBindTexture( GL_TEXTURE_2D, textures[texID] );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, t.width(), t.height(), 0, GL_RGBA8, GL_UNSIGNED_BYTE, t.bits() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

}

void ChartCloud::initializeGL(){
    glewInit( );
}

void ChartCloud::paintGL(){

    // Clear the display
    glClearColor(1,1,1,1);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    Draw( size().width(), size().height() );

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

    Keypress( event->key(), mouse_x, chartCloud_y );

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
























    /*
     * A first chart
     */



    glViewport( width/4, height/4, width/3, height/3);
    glScissor( width/4, height/4, width/3, height/3);
    glClearColor(0,1,0,0.05);
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);




    std::vector<SCI::Vex2> texCoords;
    texCoords.push_back(SCI::Vex2(0,0));
    texCoords.push_back(SCI::Vex2(1,0));
    texCoords.push_back(SCI::Vex2(1,1));
    texCoords.push_back(SCI::Vex2(0,1));

    std::vector<SCI::Vex3> chart1;

    chart1.push_back(SCI::Vex3(-20,-20,0));
    chart1.push_back(SCI::Vex3(-20,80,0));
    chart1.push_back(SCI::Vex3(80,80,0));
    chart1.push_back(SCI::Vex3(80,-20,0));


    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glBegin(GL_QUADS);
    for(int i = 0; i < (int)chart1.size(); i++){
        //glColor3f(0,0,0);
        glTexCoord2f(texCoords[i].data[0], texCoords[i].data[1]);
        glVertex3fv( chart1[i].data );
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

