#include <muView/HeartDock.h>

#include <QFileDialog>
#include <QT/QExtendedMainWindow.h>

HeartDock::HeartDock(SCI::ThirdPersonCameraControls * pView, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _smesh, Data::FiberDirectionData *_fdata, QWidget *parent) : QDockWidget(parent), render_engine( pView ) {

    vp_widget = new QSplitter( Qt::Vertical );
    sp_widget = new QSplitter( Qt::Horizontal );

    QSplitter * sp0 = new QSplitter( Qt::Horizontal );
    QSplitter * sp1 = new QSplitter( Qt::Horizontal );

    pr_widget = new ParallelCoordinates( 0 );
    tb_widget = new QTabWidget( );

    sp0->addWidget( &(render_engine.re) );
    sp0->addWidget( &(render_engine.pca) );
    sp0->addWidget( &(render_engine.reNew) );

    sp1->addWidget( &(render_engine.re2[0]) );
    sp1->addWidget( &(render_engine.re2[1]) );
    sp1->addWidget( &(render_engine.re2[2]) );

    vp_widget->addWidget( sp0 );
    vp_widget->addWidget( sp1 );
    vp_widget->addWidget( pr_widget );

    sp_widget->addWidget( vp_widget );
    sp_widget->addWidget( tb_widget );

    setWidget( sp_widget );

    QT::QControlWidget * drawBoxWidget    = new QT::QControlWidget( );
    {
        QRadioButton * draw0 = drawBoxWidget->addRadioButton( tr("Points")           );
        QRadioButton * draw1 = drawBoxWidget->addRadioButton( tr("Network")          );
        QRadioButton * draw2 = drawBoxWidget->addRadioButton( tr("Volume Rendering") );
        QRadioButton * draw3 = drawBoxWidget->addRadioButton( tr("Isosurfacing")     );
        QRadioButton * draw4 = drawBoxWidget->addRadioButton( tr("Distance Field")   );

        draw0->setChecked(true);

        connect( draw0, SIGNAL(clicked()), &(render_engine), SLOT(setDrawModePoints()) );
        connect( draw1, SIGNAL(clicked()), &(render_engine), SLOT(setDrawModeNetwork()) );
        connect( draw2, SIGNAL(clicked()), &(render_engine), SLOT(setDrawModeVolumeRendering()) );
        connect( draw3, SIGNAL(clicked()), &(render_engine), SLOT(setDrawModeIsosurfacing()) );
        connect( draw4, SIGNAL(clicked()), &(render_engine), SLOT(setDrawModeDistanceField()) );
    }

    QWidget   * colorBoxWidget    = new QWidget( );
    {
        QRadioButton * color0 = new QRadioButton( tr("Dimension Value") );
        QRadioButton * color7 = new QRadioButton( tr("Min Value") );
        QRadioButton * color1 = new QRadioButton( tr("Mean Value") );
        QRadioButton * color8 = new QRadioButton( tr("Max Value") );
        QRadioButton * color2 = new QRadioButton( tr("St Dev") );
        QRadioButton * color3 = new QRadioButton( tr("Clustering") );
        QRadioButton * color4 = new QRadioButton( tr("Isovalue") );
        QRadioButton * color5 = new QRadioButton( tr("PCA Painting") );
        QRadioButton * color6 = new QRadioButton( tr("Fiber Direction") );
        color0->setChecked(true);

        QLabel * dimension_label = new QLabel(tr("Dimension"));
        QSpinBox * dimension_spinner = new QSpinBox( );
        dimension_spinner->setRange(0,40);
        dimension_spinner->setValue(0);

        QLabel * cluster_count_label     = new QLabel(tr("Clusters"));
        QSpinBox * cluster_count_spinner = new QSpinBox( );
        cluster_count_spinner->setRange(2,40);
        cluster_count_spinner->setValue( 12 );

        QT::QControlWidget * cluster_type = new QT::QControlWidget( );
        {
            QRadioButton * ct0 = cluster_type->addRadioButton( tr("L2 Norm") );
            QRadioButton * ct1 = cluster_type->addRadioButton( tr("Pearson Correlation") );
            QRadioButton * ct2 = cluster_type->addRadioButton( tr("Histogram Difference") );
            ct0->setChecked( true );
            connect (ct0, SIGNAL(clicked()), &(render_engine), SLOT(setClusterTypeL2Norm()) );
            connect (ct1, SIGNAL(clicked()), &(render_engine), SLOT(setClusterTypePearson()) );
            connect (ct2, SIGNAL(clicked()), &(render_engine), SLOT(setClusterTypeHistogram()) );
        }

        QLabel * cluster_iteration_label     = new QLabel(tr("Iterations"));
        QSpinBox * cluster_iteration_spinner = new QSpinBox( );
        cluster_iteration_spinner->setRange(1,40);
        cluster_iteration_spinner->setValue( 5 );

        QCheckBox * cluster_histogram    = new QCheckBox( tr("Histogram") );
        cluster_histogram->setChecked( true );

        QPushButton * cluster_recalculate   = new QPushButton( tr("Recalculate") );

        QPushButton * pca_sel_color         = new QPushButton( tr("PCA: Select Paint Color") );

        QLabel   * pca_dim0_label   = new QLabel(tr("PCA X Dimension"));
        QSpinBox * pca_dim0_spinner = new QSpinBox( );
        pca_dim0_spinner->setRange(0,100);
        pca_dim0_spinner->setValue( 0 );

        QLabel   * pca_dim1_label   = new QLabel(tr("PCA Y Dimension"));
        QSpinBox * pca_dim1_spinner = new QSpinBox( );
        pca_dim1_spinner->setRange(0,100);
        pca_dim1_spinner->setValue( 1 );

        connect( color0,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeDimension()) );
        connect( color1,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeMedian()) );
        connect( color2,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeStDev()) );
        connect( color3,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeCluster()) );
        connect( color4,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeIsovalue()) );
        connect( color5,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModePCA()) );
        connect( color6,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeFibers()) );
        connect( color7,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeMin()) );
        connect( color8,                    SIGNAL(clicked()),            &(render_engine), SLOT(setColorModeMax()) );

        connect( pca_sel_color,             SIGNAL(clicked()),            &(render_engine.pca), SLOT(selectPaintColor()) );

        connect( dimension_spinner,         SIGNAL(valueChanged(int)),    &(render_engine), SLOT(setDimension(int)) );

        connect( cluster_count_spinner,     SIGNAL(valueChanged(int)),    &(render_engine), SLOT(setClusterCount(int)) );
        connect( cluster_iteration_spinner, SIGNAL(valueChanged(int)),    &(render_engine), SLOT(setClusterIterations(int)) );
        connect( cluster_recalculate,       SIGNAL(clicked()),            &(render_engine), SLOT(setClusterRecalculate()) );
        connect( cluster_histogram,         SIGNAL(clicked(bool)),        &(render_engine), SLOT(setClusterHistogram(bool)) );

        connect( color0,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color1,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color2,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color3,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color4,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color5,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color6,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color7,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( color8,                    SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );

        connect( dimension_spinner,         SIGNAL(valueChanged(int)),    pr_widget, SLOT(Reset()) );

        connect( cluster_count_spinner,     SIGNAL(valueChanged(int)),    pr_widget, SLOT(Reset()) );
        connect( cluster_iteration_spinner, SIGNAL(valueChanged(int)),    pr_widget, SLOT(Reset()) );
        connect( cluster_recalculate,       SIGNAL(clicked()),            pr_widget, SLOT(Reset()) );
        connect( cluster_histogram,         SIGNAL(clicked(bool)),        pr_widget, SLOT(Reset()) );

        connect( pca_dim0_spinner, SIGNAL(valueChanged(int)), &(render_engine.pca), SLOT(ModifyPCADim0(int)) );
        connect( pca_dim1_spinner, SIGNAL(valueChanged(int)), &(render_engine.pca), SLOT(ModifyPCADim1(int)) );

        int row = 0;
        QGridLayout * colorLayout = new QGridLayout( );
        colorLayout->addWidget( color0,                     row++, 0, 1, 3 );
        colorLayout->addWidget( dimension_label,            row,   1, 1, 1 );
        colorLayout->addWidget( dimension_spinner,          row++, 2, 1, 1 );
        colorLayout->addWidget( color1,                     row++, 0, 1, 3 );
        colorLayout->addWidget( color7,                     row++, 0, 1, 3 );
        colorLayout->addWidget( color8,                     row++, 0, 1, 3 );
        colorLayout->addWidget( color2,                     row++, 0, 1, 3 );
        colorLayout->addWidget( color3,                     row++, 0, 1, 3 );
        colorLayout->addWidget( cluster_histogram,          row++, 1, 1, 2 );
        colorLayout->addWidget( cluster_count_label,        row,   1, 1, 1 );
        colorLayout->addWidget( cluster_count_spinner,      row++, 2, 1, 1 );
        colorLayout->addWidget( cluster_type,               row++, 1, 1, 1 );
        colorLayout->addWidget( cluster_iteration_label,    row,   1, 1, 1 );
        colorLayout->addWidget( cluster_iteration_spinner,  row++, 2, 1, 1 );
        colorLayout->addWidget( cluster_recalculate,        row++, 1, 1, 2 );
        colorLayout->addWidget( color4,                     row++, 0, 1, 3 );
        colorLayout->addWidget( color5,                     row++, 0, 1, 3 );
        colorLayout->addWidget( pca_sel_color,              row++, 0, 1, 3 );
        colorLayout->addWidget( color6,                     row++, 0, 1, 3 );
        colorLayout->addWidget( pca_dim0_label,             row,   1, 1, 1 );
        colorLayout->addWidget( pca_dim0_spinner,           row++, 2, 1, 1 );
        colorLayout->addWidget( pca_dim1_label,             row,   1, 1, 1 );
        colorLayout->addWidget( pca_dim1_spinner,           row++, 2, 1, 1 );

        colorLayout->setRowStretch( row, 1 );

        colorBoxWidget->setLayout( colorLayout );
    }



    QWidget   * isoBoxWidget    = new QWidget( );
    {

        QCheckBox * show_d_iso    = new QCheckBox( tr("Dimension Isosurface") );
        QCheckBox * show_min_iso  = new QCheckBox( tr("Minimum Isosurface") );
        QCheckBox * show_mean_iso = new QCheckBox( tr("Mean Isosurface") );
        QCheckBox * show_max_iso  = new QCheckBox( tr("Maximum Isosurface") );
        QLabel * iso_label        = new QLabel(tr("Iso value"));
        QDoubleSpinBox * iso_spinner = new QDoubleSpinBox( );
        iso_spinner->setRange(-15,15);
        iso_spinner->setValue(0);
        QPushButton * volume       = new QPushButton( tr("Volume") );

        connect( show_d_iso,    SIGNAL(clicked(bool)),        &(render_engine), SLOT(setDimIsosurface(bool)) );
        connect( show_min_iso,  SIGNAL(clicked(bool)),        &(render_engine), SLOT(setMinIsosurface(bool)) );
        connect( show_mean_iso, SIGNAL(clicked(bool)),        &(render_engine), SLOT(setMeanIsosurface(bool)) );
        connect( show_max_iso,  SIGNAL(clicked(bool)),        &(render_engine), SLOT(setMaxIsosurface(bool)) );
        connect( iso_spinner,   SIGNAL(valueChanged(double)), &(render_engine), SLOT(setIsovalue(double)) );
        connect( iso_spinner,   SIGNAL(valueChanged(double)), pr_widget, SLOT(Reset()) );
        connect( volume,        SIGNAL(clicked()), &(render_engine), SLOT(calculateSubVolume()) );


        int row = 0;
        QGridLayout * isoLayout = new QGridLayout( );
        isoLayout->addWidget( show_d_iso,                 row++, 1, 1, 3 );
        isoLayout->addWidget( show_min_iso,               row++, 1, 1, 3 );
        isoLayout->addWidget( show_mean_iso,              row++, 1, 1, 3 );
        isoLayout->addWidget( show_max_iso,               row++, 1, 1, 3 );
        isoLayout->addWidget( iso_label,                  row,   1, 1, 1 );
        isoLayout->addWidget( iso_spinner,                row++, 2, 1, 1 );
        isoLayout->addWidget( volume,                     row++, 1, 1, 3 );
        isoLayout->setRowStretch( row, 1 );

        isoBoxWidget->setLayout( isoLayout );
    }


    QWidget   * clipBoxWidget    = new QWidget( );
    {
        QCheckBox * e_clipX    = new QCheckBox( tr("Clip X") );
        QCheckBox * e_clipY    = new QCheckBox( tr("Clip Y") );
        QCheckBox * e_clipZ    = new QCheckBox( tr("Clip Z") );
        QCheckBox * f_clipX    = new QCheckBox( tr("flip") );
        QCheckBox * f_clipY    = new QCheckBox( tr("flip") );
        QCheckBox * f_clipZ    = new QCheckBox( tr("flip") );
        QDoubleSpinBox * v_clipX = new QDoubleSpinBox( );
        QDoubleSpinBox * v_clipY = new QDoubleSpinBox( );
        QDoubleSpinBox * v_clipZ = new QDoubleSpinBox( );
        v_clipX->setRange(-15,15);
        v_clipX->setSingleStep(0.05f);
        v_clipX->setValue(0);
        v_clipY->setRange(-15,15);
        v_clipY->setSingleStep(0.05f);
        v_clipY->setValue(0);
        v_clipZ->setRange(-15,15);
        v_clipZ->setValue(0);
        v_clipZ->setSingleStep(0.05f);

        connect( e_clipX, SIGNAL(clicked(bool)), &(render_engine), SLOT(setClipXEnable(bool)) );
        connect( e_clipY, SIGNAL(clicked(bool)), &(render_engine), SLOT(setClipYEnable(bool)) );
        connect( e_clipZ, SIGNAL(clicked(bool)), &(render_engine), SLOT(setClipZEnable(bool)) );

        connect( v_clipX, SIGNAL(valueChanged(double)), &(render_engine), SLOT(setClipXVal(double)) );
        connect( v_clipY, SIGNAL(valueChanged(double)), &(render_engine), SLOT(setClipYVal(double)) );
        connect( v_clipZ, SIGNAL(valueChanged(double)), &(render_engine), SLOT(setClipZVal(double)) );

        connect( f_clipX, SIGNAL(clicked()), &(render_engine), SLOT(setClipXFlip()) );
        connect( f_clipY, SIGNAL(clicked()), &(render_engine), SLOT(setClipYFlip()) );
        connect( f_clipZ, SIGNAL(clicked()), &(render_engine), SLOT(setClipZFlip()) );

        int row = 0;
        QGridLayout * clipLayout = new QGridLayout( );
        clipLayout->addWidget( e_clipX, row,   0, 1, 1 );
        clipLayout->addWidget( v_clipX, row,   1, 1, 1 );
        clipLayout->addWidget( f_clipX, row++, 2, 1, 1 );
        clipLayout->addWidget( e_clipY, row,   0, 1, 1 );
        clipLayout->addWidget( v_clipY, row,   1, 1, 1 );
        clipLayout->addWidget( f_clipY, row++, 2, 1, 1 );
        clipLayout->addWidget( e_clipZ, row,   0, 1, 1 );
        clipLayout->addWidget( v_clipZ, row,   1, 1, 1 );
        clipLayout->addWidget( f_clipZ, row++, 2, 1, 1 );
        clipLayout->setRowStretch( row, 1 );

        clipBoxWidget->setLayout( clipLayout );
    }


    tb_widget->setTabPosition( tb_widget->West );
    tb_widget->addTab(  drawBoxWidget, tr( "Draw Mode" ) );
    tb_widget->addTab( colorBoxWidget, tr( "Color Mode" ) );
    tb_widget->addTab(   isoBoxWidget, tr( "Isosurfacing" ) );
    tb_widget->addTab(  clipBoxWidget, tr( "Clip Planes" ) );


    dfield = 0;
    pdata  = 0;
    pmesh  = _pmesh;
    tdata  = _smesh;
    fdata  = _fdata;

    render_engine.SetParallelCoordinateView( pr_widget );
    render_engine.SetData( pdata, pmesh, tdata );
    render_engine.SetFiberData( fdata );

}

void HeartDock::SetPointData( Data::PointData * _pdata ){
    if( pdata != _pdata ){
        if(pdata) delete pdata;
        pdata = _pdata;
        render_engine.SetData( pdata, pmesh, tdata );
        setWindowTitle( tr(pdata->GetFilename().c_str()) );
    }
}

void HeartDock::SetDistanceFieldData( Data::DistanceFieldSet * _dfield ){
    if( dfield != _dfield ){
        if(dfield) delete dfield;
        dfield = _dfield;
        render_engine.SetDistanceFieldData( dfield );
    }
}

void HeartDock::AddImportedMesh( Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata ){
    render_engine.AddImportedMesh( _pmesh,_tdata );
}



Data::Mesh::PointMesh * HeartDock::OpenPointMesh( ){

    QString fname = QT::QExtendedMainWindow::openDialog( tr("Load a Point Mesh"), QString("Any (*.point *.point.gz *.pts);; Point File (*.point);;Compressed Point File (*.point.gz);; PTS File (*.pts)") );

    // Loading a point data file
    if( fname.endsWith(".point") || fname.endsWith(".point.gz") ){
        return new Data::Mesh::PointMesh( fname.toLocal8Bit().data(), true, fname.endsWith(".gz") );
    }
    // Loading raw pts data file
    else if( fname.endsWith(".pts") ){
        return new Data::Mesh::PointMesh( fname.toLocal8Bit().data(), false, false );
    }

    return 0;

}

Data::DistanceFieldSet * HeartDock::OpenDistanceField( ){

    QString fname = QT::QExtendedMainWindow::openDialog( tr("Load a Distance Field"), QString("Distance Field File (*.df *.dfield)") );

    if( fname.endsWith(".dfield") || fname.endsWith(".df") ){
        return new Data::DistanceFieldSet( fname.toLocal8Bit().data() );
    }

    return 0;

}


Data::Mesh::SolidMesh * HeartDock::OpenSolidMesh( ){

    QString mesh_name = QT::QExtendedMainWindow::openDialog( tr("Load an Associated Mesh"), QString("Any (*.tet *.hex *.btet *.btet.gz *.bhex);; Tet File (*.tet);; Binary Tets File (*.btet *.btet.gz);; Hex File (*.hex);; Binary Hex File (*.bhex)") );

    if( mesh_name.endsWith(".btet") || mesh_name.endsWith(".btet.gz") || mesh_name.endsWith(".tet") ){
        return new Data::Mesh::TetMesh( mesh_name.toLocal8Bit().data() );
    }

    if( mesh_name.endsWith(".bhex") || mesh_name.endsWith(".hex") ){
        return new Data::Mesh::HexMesh( mesh_name.toLocal8Bit().data() );
    }

    return 0;

}

Data::PointData * HeartDock::OpenPointData( Data::PointData * pdata ){

    QStringList fname_list = QT::QExtendedMainWindow::openListDialog( tr("Load Data Files"), QString("All Data Files (*.pdata *.txt *.sol);;Point Data File (*.pdata);;Text Data Files (*.txt);;Solution File (*.sol)") );

    for(int i = 0; i < fname_list.size(); i++){

        Data::PointData * ptmp0 = pdata;
        Data::PointData * ptmp1 = new Data::PointData( fname_list.at(i).toLocal8Bit().data(), fname_list.at(i).endsWith(".pdata") );

        if(pdata == 0) {
            pdata = ptmp1;
        }
        else {
            pdata = new Data::PointData( *ptmp0, *ptmp1 );
            delete ptmp0;
            delete ptmp1;
        }
    }

    return pdata;

}

Data::FiberDirectionData * HeartDock::OpenFiberData( ){
    QString fibs_name = QT::QExtendedMainWindow::openDialog( tr("Load Fiber Data"), QString("Any (*.txt *.fibs);; Fiber File (*.txt);; Binary Fiber File (*.fibs)") );

    if( fibs_name.endsWith(".txt") || fibs_name.endsWith(".fibs") ){
        return new Data::FiberDirectionData( fibs_name.toLocal8Bit().data(), fibs_name.endsWith(".fibs") );
    }

    return 0;
}


