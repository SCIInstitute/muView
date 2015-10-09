#include <GL/glew.h>
#include <MainWidget.h>


MainWidget::MainWidget( Dialog::Metadata *_data, QMenuBar *_menu_bar, QWidget *parent) : QGLWidget( QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer | QGL::Rgba | QGL::AlphaChannel | QGL::DirectRendering | QGL::SampleBuffers), parent ), menu_bar(_menu_bar) {

    data = _data->GetProxyData();

    if( data->conn.size() > 0 && data->vert.size() > 0 ){
        draw_data = new Drawable::MeshDisplay();
    }
    if( data->conn.size() == 0 && data->vert.size() > 0 ){
        draw_data = new Drawable::PointDisplay();
    }
    if( data->conn.size() == 0 && data->vert.size() == 0 ){
        draw_data = new Drawable::VolumeDisplay( );
    }

    // Setup menu for Dimensionality Reduction Options
    dr_method_menu = new QMenu("Dimensionality Reduction");
    {

        dr_method_group = new QActionGroup(this);
        {
            QAction* action;

            for(int i = 0; i < (int)_data->GetMethodCount(); i++){
                action = new QAction(tr(_data->GetMethod(i).c_str()), this);
                action->setData( QVariant( i ) );
                action->setCheckable(true);
                if(i==0) action->setChecked(true);
                dr_method_menu->addAction(action);
                dr_method_group->addAction(action);
            }
            connect( dr_method_group, SIGNAL(triggered( QAction * )), &prov, SLOT(Push(QAction*)) );

        }
    }
    dr_method_menu_action = 0;

    view_menu = new QMenu("View");
    {
        view_reset = view_menu->addAction(tr("Reset"));
        connect( view_reset, SIGNAL(triggered()), draw_data, SLOT(ResetView()) );
    }
    view_menu_action = 0;

    mouse_button = Drawable::DrawableObject::NoButton;
    mouse_x = 0;
    mouse_y = 0;

    setMouseTracking(true);

}

MainWidget::~MainWidget(){

}

void MainWidget::hideEvent ( QHideEvent * event ){
    // Window is hidden, detach menus
    DetachMenus();
}

void MainWidget::showEvent ( QShowEvent * event ){
    // Window is appears, attach menus
    AttachMenus();
}


void MainWidget::AttachMenus( ){
    // Code to attach menus
    if(dr_method_menu_action==0){
        dr_method_menu_action = menu_bar->addMenu( dr_method_menu );
    }
    if(view_menu_action==0){
        view_menu_action = menu_bar->addMenu( view_menu );
    }
}

void MainWidget::DetachMenus( ){
    // Code to detatch menus
    if(dr_method_menu_action!=0){
        menu_bar->removeAction( dr_method_menu_action );
        dr_method_menu_action = 0;
    }
    if(view_menu_action!=0){
        menu_bar->removeAction( view_menu_action );
        view_menu_action = 0;
    }
}

QSize MainWidget::minimumSizeHint() const {
    // Minimum gl canvas size
    return QSize(50, 50);
}

QSize MainWidget::sizeHint() const {
    // Desired gl canvas size
    return QSize(1280, 720);
}

void MainWidget::initializeGL(){
    // Initialization code
    glewInit( );

    prov.Initialize( *data );
    if( dynamic_cast<Drawable::VolumeDisplay*>(draw_data) )
        dynamic_cast<Drawable::VolumeDisplay*>(draw_data)->Initialize( *data, prov );
    if( dynamic_cast<Drawable::MeshDisplay*>(draw_data) )
        dynamic_cast<Drawable::MeshDisplay*>(draw_data)->Initialize( *data, prov );
    if( dynamic_cast<Drawable::PointDisplay*>(draw_data) )
        dynamic_cast<Drawable::PointDisplay*>(draw_data)->Initialize( *data, prov );
    draw_dr.Initialize( prov, *dynamic_cast<Data::BasicData*>(data) );

}

void MainWidget::paintGL(){

    // Clear the display
    glClearColor(1,1,1,1);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    int w = size().width();
    int h = size().height();

    int vol_w = 3*w/4;
    int vol_h = h;

    int dr_w = w / 4;
    int dr_h = dr_w;

    // Draw the volume
    draw_data->SetShape( 0,0,vol_w,vol_h );
    draw_data->Draw( );

    // Draw Provenance View
    prov.SetShape( w-dr_w, dr_h, dr_w, h-dr_h );
    prov.Draw( );

    // Draw DR
    draw_dr.SetShape( w-dr_w, 0, dr_w, dr_h );
    draw_dr.Draw( );

    update();

}

void MainWidget::mouseDoubleClickEvent ( QMouseEvent * event ){

    draw_dr.ClearSelection();
    prov.ClearSelection();
    draw_data->ClearSelection();

    int x = event->x();
    int y = size().height()-event->y()-1;

    bool event_handled = false;

    if( !event_handled ){ event_handled =  draw_dr.MouseClick( event->button(), Drawable::DrawableObject::Double, x, y ); }
    if( !event_handled ){ event_handled =     prov.MouseClick( event->button(), Drawable::DrawableObject::Double, x, y ); }
    if( !event_handled ){ event_handled = draw_data->MouseClick( event->button(), Drawable::DrawableObject::Double, x, y ); }

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void MainWidget::mouseMoveEvent ( QMouseEvent * event ){

    draw_dr.ClearSelection();
    prov.ClearSelection();
    draw_data->ClearSelection();

    int x = event->x();
    int y = size().height()-event->y()-1;

    bool event_handled = false;

    if( !event_handled ){ event_handled =  draw_dr.MouseMotion( mouse_button, x, x-mouse_x, y, y-mouse_y ); }
    if( !event_handled ){ event_handled =     prov.MouseMotion( mouse_button, x, x-mouse_x, y, y-mouse_y ); }
    if( !event_handled ){ event_handled = draw_data->MouseMotion( mouse_button, x, x-mouse_x, y, y-mouse_y ); }

    mouse_x = x;
    mouse_y = y;

}

void MainWidget::mousePressEvent ( QMouseEvent * event ){

    draw_dr.ClearSelection();
    prov.ClearSelection();
    draw_data->ClearSelection();

    int x = event->x();
    int y = size().height()-event->y()-1;

    bool event_handled = false;

    if( !event_handled ){ event_handled =  draw_dr.MouseClick( event->button(), Drawable::DrawableObject::Down, x, y ); }
    if( !event_handled ){ event_handled =     prov.MouseClick( event->button(), Drawable::DrawableObject::Down, x, y ); }
    if( !event_handled ){ event_handled = draw_data->MouseClick( event->button(), Drawable::DrawableObject::Down, x, y ); }

    mouse_button = event->button();
    mouse_x = x;
    mouse_y = y;

}

void MainWidget::mouseReleaseEvent ( QMouseEvent * event ){

    draw_dr.ClearSelection();
    prov.ClearSelection();
    draw_data->ClearSelection();

    int x = event->x();
    int y = size().height()-event->y()-1;

    bool event_handled = false;

    if( !event_handled ){ event_handled =  draw_dr.MouseClick( event->button(), Drawable::DrawableObject::Up, x, y ); }
    if( !event_handled ){ event_handled =     prov.MouseClick( event->button(), Drawable::DrawableObject::Up, x, y ); }
    if( !event_handled ){ event_handled = draw_data->MouseClick( event->button(), Drawable::DrawableObject::Up, x, y ); }

    mouse_button = Drawable::DrawableObject::NoButton;
    mouse_x = x;
    mouse_y = y;

}

void MainWidget::keyPressEvent ( QKeyEvent * event ){

    bool event_handled = false;

    if( !event_handled ){ event_handled =  draw_dr.Keypress( event->key(), mouse_x, mouse_y ); }
    if( !event_handled ){ event_handled =     prov.Keypress( event->key(), mouse_x, mouse_y ); }
    if( !event_handled ){ event_handled = draw_data->Keypress( event->key(), mouse_x, mouse_y ); }

}

void MainWidget::keyReleaseEvent ( QKeyEvent * event ){

    bool event_handled = false;

    if( !event_handled ){ event_handled =  draw_dr.Keypress( event->key(), mouse_x, mouse_y ); }
    if( !event_handled ){ event_handled =     prov.Keypress( event->key(), mouse_x, mouse_y ); }
    if( !event_handled ){ event_handled = draw_data->Keypress( event->key(), mouse_x, mouse_y ); }

}
