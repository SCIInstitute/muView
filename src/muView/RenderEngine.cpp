#include <muView/RenderEngine.h>

#include <iostream>

#include <QMouseEvent>

#include <SCI/Colormap.h>

RenderEngine::RenderEngine(SCI::ThirdPersonCameraControls *_pView) {

    re.pView = _pView;


    draw_mode  =  0;
    color_mode =  0;
    color_dim  =  0;
    clusterN   = 12;
    clusterI   =  5;
    cluster_histogram = true;
    pca_color_dim  =  1;
    isoval    =  0;
    view_dim_iso  = false;
    view_min_iso  = false;
    view_mean_iso = false;
    view_max_iso  = false;
    view_fibers   = false;
    fiber_length = 0.2f;

    tdata  = 0;
    pdata  = 0;
    fdata  = 0;
    pmesh  = 0;
    dfield = 0;

    //font.Load( "../fonts/arial.font" );
    font.Load( ":/fonts/arial.font" );


    chartCloud.pdata = pdata;
    chartCloud.pmesh = pmesh;
    chartCloud.tdata = tdata;
    chartCloud.NeedUpdate();
    chartCloud.draw_mode = draw_mode;
    chartCloud.iso_points = &iso_points;
    chartCloud.iso_tris = &iso_tris;
    chartCloud.iso_tets = &iso_tets;
    chartCloud.iso_hexs = &iso_hexs;
    chartCloud.iso_color = &iso_color;
    chartCloud.df_points = &df_points;
    chartCloud.df_tris = &df_tris;
    chartCloud.df_color = &df_color;
    //chartCloud.edge_data = &edge_data;
    //chartCloud.edge_mesh = &edge_mesh;
    chartCloud.cluster = &cluster;
    //chartCloud.vox_assoc = &vox_assoc;
    chartCloud.colormap = &colormap;
    chartCloud.seq_cmap = &seq_cmap;
    chartCloud.cat_cmap = &cat_cmap;
    chartCloud.draw_mode = draw_mode;
    chartCloud.color_mode = color_mode;
    chartCloud.font = &font;
    chartCloud.cluster_histogram = cluster_histogram;


    re.pdata = pdata;
    re.pmesh = pmesh;
    re.tdata = tdata;
    re.NeedUpdate();
    re.draw_mode = draw_mode;
    re.iso_points = &iso_points;
    re.iso_tris = &iso_tris;
    re.iso_tets = &iso_tets;
    re.iso_hexs = &iso_hexs;
    re.iso_color = &iso_color;
    re.df_points = &df_points;
    re.df_tris = &df_tris;
    re.df_color = &df_color;
    //re.edge_data = &edge_data;
    //re.edge_mesh = &edge_mesh;
    re.cluster = &cluster;
    //re.vox_assoc = &vox_assoc;
    re.colormap = &colormap;
    re.seq_cmap = &seq_cmap;
    re.cat_cmap = &cat_cmap;
    re.draw_mode = draw_mode;
    re.color_mode = color_mode;
    re.font = &font;
    re.cluster_histogram = cluster_histogram;

    pca.font = &font;

    for(int i = 0; i < 3; i++){
        re.pln[i] = &(re2[i].plane);
        re.pln_color[i] = &(re2[i].pln_color);

        chartCloud.pln[i] = &(re2[i].plane);
        chartCloud.pln_color[i] = &(re2[i].pln_color);

        re2[i].font = &font;
        connect( &(re2[i]), SIGNAL(Updated(RenderEngine2D*)), this, SLOT(UpdateRenderEngine2DColor(RenderEngine2D*)) );
    }
    re2[0].pln_color = SCI::Vex3(127, 201, 127)/255.0f;
    re2[1].pln_color = SCI::Vex3(190, 174, 212)/255.0f;
    re2[2].pln_color = SCI::Vex3(253, 192, 134)/255.0f;

    connect( &cluster, SIGNAL(IterationComplete()), this, SLOT(Recalculate()) );
}




void RenderEngine::setDrawModePoints( ){          draw_mode  = 0; Recalculate(); }
void RenderEngine::setDrawModeVolumeRendering( ){ draw_mode  = 1; Recalculate(); }
void RenderEngine::setDrawModeIsosurfacing( ){    draw_mode  = 2; Recalculate(); }
void RenderEngine::setDrawModeDistanceField( ){   draw_mode  = 3; Recalculate(); }
void RenderEngine::setDrawModeNetwork( ){         draw_mode  = 4; Recalculate(); }

void RenderEngine::setColorModeDimension( ){      color_mode = 0; Recalculate(); }
void RenderEngine::setColorModeMin( ){            color_mode = 7; Recalculate(); }
void RenderEngine::setColorModeMax( ){            color_mode = 8; Recalculate(); }
void RenderEngine::setColorModeMedian( ){         color_mode = 1; Recalculate(); }
void RenderEngine::setColorModeStDev( ){          color_mode = 2; Recalculate(); }
void RenderEngine::setColorModeCluster( ){        color_mode = 3; Recalculate(); }
void RenderEngine::setColorModeIsovalue( ){       color_mode = 4; Recalculate(); }
void RenderEngine::setColorModePCA( ){            color_mode = 5; Recalculate(); }
void RenderEngine::setColorModeFibers( ){         color_mode = 6; Recalculate(); }
void RenderEngine::setColorModePCAcolor( ){       color_mode = 9;
                                                  Recalculate(); }

void RenderEngine::setDimension( int v ){         color_dim  = v; Recalculate(); }
void RenderEngine::setPrincipalComponentNumber( int p ){         pca_color_dim  = p; Recalculate(); }

void RenderEngine::setClusterCount( int v ){      clusterN   = v;                }
void RenderEngine::setClusterIterations( int v ){ clusterI   = v;                }
void RenderEngine::setClusterRecalculate( ){                      Recalculate(); }
void RenderEngine::setClusterHistogram( bool v ){ cluster_histogram = v; }
void RenderEngine::setClusterTypeL2Norm( ){       cluster_type = KMeansClustering::KM_L2Norm;  }
void RenderEngine::setClusterTypePearson( ){      cluster_type = KMeansClustering::KM_Pearson; }
void RenderEngine::setClusterTypeHistogram( ){    /*cluster_type = KMeansClustering::KM_Histogram;*/ }


//void RenderEngine::setIsotetrahedron( bool v ){   view_iso_tet  = v; Recalculate(); }
void RenderEngine::setDimIsosurface( bool v ){    view_dim_iso  = v; Recalculate(); }
void RenderEngine::setMinIsosurface( bool v ){    view_min_iso  = v; Recalculate(); }
void RenderEngine::setMeanIsosurface( bool v ){   view_mean_iso = v; Recalculate(); }
void RenderEngine::setMaxIsosurface( bool v ){    view_max_iso  = v; Recalculate(); }
void RenderEngine::setIsovalue( double v ){       isoval        = v; Recalculate(); }

void RenderEngine::setClipXVal( double v ){ re.clpX[3] = re.clpX[0] * v; chartCloud.clpX[3] = chartCloud.clpX[0] * v; }
void RenderEngine::setClipYVal( double v ){ re.clpY[3] = re.clpY[1] * v; chartCloud.clpY[3] = chartCloud.clpY[1] * v;}
void RenderEngine::setClipZVal( double v ){ re.clpZ[3] = re.clpZ[2] * v; chartCloud.clpZ[3] = chartCloud.clpZ[2] * v;}

void RenderEngine::setClipXFlip( ){ re.clpX[0] *= -1.0f; chartCloud.clpX[0] *= -1.0f;}
void RenderEngine::setClipYFlip( ){ re.clpY[1] *= -1.0f; chartCloud.clpY[1] *= -1.0f;}
void RenderEngine::setClipZFlip( ){ re.clpZ[2] *= -1.0f; chartCloud.clpZ[2] *= -1.0f;}

void RenderEngine::setClipXEnable( bool v ){ re.useClipX = v; chartCloud.useClipX = v;}
void RenderEngine::setClipYEnable( bool v ){ re.useClipY = v; chartCloud.useClipY = v;}
void RenderEngine::setClipZEnable( bool v ){ re.useClipZ = v; chartCloud.useClipZ = v;}

void RenderEngine::setFiberDirection( bool v ){   view_fibers   = v; }
void RenderEngine::setFiberLength( double v ){     fiber_length  = v; }

void RenderEngine::calculateSubVolume(){
    iso_tets.Clear();
    iso_hexs.Clear();

    tdata->ExtractIsotetrahedron( iso_tets, *pdata, isoval );
    tdata->ExtractIsohexahedron( iso_hexs, *pdata, isoval );
    float v  = iso_tets.Volume( *pmesh ) + iso_hexs.Volume( *pmesh );
    float tv = tdata->Volume(*pmesh);
    std::cout << "RenderEngine: " << "Volume " << v << " units (microliters?) of total volume " << tv << " units (" << v/tv*100 << "%)" << std::endl << std::flush;
}

void RenderEngine::SetFiberData(Data::FiberDirectionData *_fdata){
    fdata = _fdata;
}

void RenderEngine::SetDistanceFieldData( Data::DistanceFieldSet * _dfield ){
    dfield = _dfield;
    df_tris.clear();
    df_points.Clear();
}

void RenderEngine::AddImportedMesh( Data::Mesh::PointMesh *pmesh, Data::Mesh::SolidMesh *tdata ){
    re.addl_points.push_back( pmesh );
    re.addl_solid.push_back( tdata );

    chartCloud.addl_points.push_back( pmesh );
    chartCloud.addl_solid.push_back( tdata );
}

void RenderEngine::SetData( Data::PointData * _pdata, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata  ){

    if( _pdata == 0 || _pmesh == 0 || _tdata == 0 ) return;

    pdata   = _pdata;
    tdata   = _tdata;
    pmesh   = _pmesh;

    SCI::Vex3 bbmid = pmesh->bb.GetCenter();
    float     siz   = pmesh->bb.GetMaximumDimensionSize();

    tdata->BuildSpacePartition( *pmesh );
    tdata->tri_mesh.FixNormals( *pmesh );

    std::cout << "RenderEngine: " << "Finding clusters" << std::endl << std::flush;
    cluster_type = KMeansClustering::KM_L2Norm;
    cluster.Initialize( pdata, 1, 1, cluster_type );
    //std::cout << "Estimated clusters: " << (int)sqrt( (double)pdata->GetElementCount()/2.0) << std::endl << std::flush;
    std::cout << "Estimated clusters: " << (int)(log( (double)pdata->GetElementCount()/2.0) / log(2.0)) << std::endl << std::flush;

    /*
    std::cout << "RenderEngine: " << "Extracting Edges" << std::endl << std::flush;
    edge_mesh.clear();
    tdata->ExtractEdges( edge_mesh );
    edge_data.Resize( (int)edge_mesh.size(), pdata->GetDimension() );
    for(int i = 0; i < (int)edge_mesh.size(); i++){
        edge_data.SetElement( i, SCI::fabsf( pdata->GetElement(edge_mesh[i].v0) - pdata->GetElement(edge_mesh[i].v1) ) );
    }
    */

    re2[0].SetData( pdata, pmesh, tdata, SCI::Vex4(1,0,0,-pmesh->bb.GetCenter().x), SCI::Vex4( 90, 0, 1, 0 ) );
    re2[1].SetData( pdata, pmesh, tdata, SCI::Vex4(0,1,0,-pmesh->bb.GetCenter().y), SCI::Vex4(-90, 1, 0, 0 ) );
    re2[2].SetData( pdata, pmesh, tdata, SCI::Vex4(0,0,1,-pmesh->bb.GetCenter().z), SCI::Vex4(  0, 1, 0, 0 ) );
    pca.SetData( pdata, pmesh, tdata, &colormap );
    pcaAxis.SetData( pdata, pmesh, tdata, &colormap );

    std::cout << "RenderEngine: " << "Recalculating colors" << std::endl << std::flush;
    Recalculate();

}

void RenderEngine::UpdateRenderEngine2DColor( RenderEngine2D * re ){
    re->iso_lines.clear();
    re->iso_verts.Clear();
    if( color_mode == 0 ){
        re->colormap.SetByDataDimension( *(re->ptmp), seq_cmap, color_dim );
        re->ttmp.ExtractIsolineByDataDimension( re->iso_lines, re->iso_verts, re->vtmp, *(re->ptmp), color_dim, 1 );
    }
    if( color_mode == 1 ){
        re->colormap.SetByDataMean( *(re->ptmp), seq_cmap );
        re->ttmp.ExtractIsolineByMeanValue( re->iso_lines, re->iso_verts, re->vtmp, *(re->ptmp), 1 );
    }
    if( color_mode == 2 ){
        re->colormap.SetByDataStdev( *(re->ptmp), seq_cmap );
        re->ttmp.ExtractIsolineByStdevValue( re->iso_lines, re->iso_verts, re->vtmp, *(re->ptmp), 1 );
    }
    if( color_mode == 3 ){
        re->colormap.clear();
        re->colormap.SetColorPerVertex();
        for(int i = 0; i < re->ptmp->GetElementCount(); i++){
            int cid = cluster.GetClusterID( re->ptmp->GetElement(i) );
            re->colormap.push_back( cat_cmap.GetColor( cid ).UIntColor() );
        }
    }
    if( color_mode == 4 ){
        SCI::Vex4 mincol = SCI::Vex4( 213,  62,  79, 10 ) / 255.0f;
        SCI::Vex4 maxcol = SCI::Vex4(  50, 136, 189, 10 ) / 255.0f;
        re->colormap.SetByDataIsoRange( *(re->ptmp), seq_cmap, mincol, maxcol, isoval );
    }

    if( color_mode == 9){

        //TODO implement
        //re->colormap.setByPCAcolor( *(re->ptmp), seq_cmap, pca_color_dim );
        //re->ttmp.ExtractIsolineByDataDimension( re->iso_lines, re->iso_verts, re->vtmp, *(re->ptmp), pca_color_dim, 1 );


        pca_color_dim = pca_color_dim % 3;
        SCI::Vex4 color;
        switch(pca_color_dim){
            case 0:
                color = SCI::Vex4(1,0,0,1);
                break;
            case 1:
                color = SCI::Vex4(0,1,0,1);
                break;
            case 2:
                color = SCI::Vex4(0,0,1,1);
                break;
        }

        int subsetSize = pcaAxis.getSubsetSize();
        double * colorAxis = new double[subsetSize];
        colorAxis = pcaAxis.getPrincipalComponent(0);

        std::vector<int> subsetIndices;
        pcaAxis.getSubsetIndices(subsetIndices);




        re->colormap.clear();
        re->colormap.SetColorPerVertex();
        for(int i = 0; i < re->ptmp->GetElementCount(); i++){
            int cid = cluster.GetClusterID( re->ptmp->GetElement(i) );
            re->colormap.push_back( color.UIntColor() );
        }
    }

    re->update();
}


void RenderEngine::Recalculate( ){

    pca.DisablePainting();

    if( draw_mode == 0 || draw_mode == 1 ){

        if( pdata == 0 ) return;

        if( color_mode == 0 ){
            seq_cmap.LoadDefaultMapRed();
            colormap.SetByDataDimension( *pdata, seq_cmap, color_dim );
        }
        if( color_mode == 1 ){
            seq_cmap.LoadDefaultMapRed();
            colormap.SetByDataMean( *pdata, seq_cmap );
        }
        if( color_mode == 7 ){
            seq_cmap.LoadDefaultMapRed();
            colormap.SetByDataMin( *pdata, seq_cmap );
        }
        if( color_mode == 8 ){
            seq_cmap.LoadDefaultMapRed();
            colormap.SetByDataMax( *pdata, seq_cmap );
        }
        if( color_mode == 2 ){
            seq_cmap.LoadDefaultMapBlue();
            colormap.SetByDataStdev( *pdata, seq_cmap );
        }
        if( color_mode == 3 ){
            cat_cmap.LoadDefaultMap();
            if( cluster.GetClusterCount() != clusterN || cluster.GetClusterType() != cluster_type ){
                cluster.Initialize( pdata, clusterN, clusterI, cluster_type );
            }
        }
        if( color_mode == 4 ){
            seq_cmap.Clear();
            SCI::Vex4 mincol = SCI::Vex4( 213,  62,  79, 10 ) / 255.0f;
            seq_cmap.Insert( SCI::Vex4( 252, 141,  89, 50 ) / 255.0f );
            seq_cmap.Insert( SCI::Vex4( 254, 224, 139, 50 ) / 255.0f );
            seq_cmap.Insert( SCI::Vex4( 255, 255, 191, 50 ) / 255.0f );
            seq_cmap.Insert( SCI::Vex4( 230, 245, 152, 50 ) / 255.0f );
            seq_cmap.Insert( SCI::Vex4( 153, 213, 148, 50 ) / 255.0f );
            SCI::Vex4 maxcol = SCI::Vex4(  50, 136, 189, 10 ) / 255.0f;
            colormap.SetByDataIsoRange( *pdata, seq_cmap, mincol, maxcol, isoval );
        }
        if( color_mode == 5 ){
            pca.EnablePainting();
        }
        if(color_mode == 9){

            pca_color_dim = pca_color_dim % 3;

            // what does that do?
            seq_cmap.LoadDefaultMapBlue();

            //TODO implement
            //re->colormap.setByPCA(pca_color_dim);

            colormap.clear();
            colormap.SetColorPerVertex();
            SCI::Vex4 color;
            switch(pca_color_dim){
                case 0:
                    color = SCI::Vex4(1,0,0,1);
                    break;
                case 1:
                    color = SCI::Vex4(0,1,0,1);
                    break;
                case 2:
                    color = SCI::Vex4(0,0,1,1);
                    break;
            }
            for(int i = 0; i < pdata->GetElementCount(); i++){
                colormap.push_back( color.UIntColor() );
            }

            // TODO
            // need new Dimensionality reduction need different pca
        }
        if( color_mode == 6 ){

            if( fdata ){

                std::vector< SCI::Vex3 > dirs( pdata->GetElementCount(), SCI::Vex3() );
                for(int i = 0; i < tdata->GetElementCount(); i++){
                    std::vector< int > ind;
                    tdata->ElementIndices(i,ind);
                    for(int j = 0; j < (int)ind.size(); j++){
                        dirs[ind[j]] += fdata->fibs[ i ].UnitVector();
                    }
                }
                colormap.clear();
                for(int i = 0; i < (int)dirs.size(); i++){
                    SCI::Vex3 cd = dirs[i].UnitVector();
                    float H;
                    if( fabsf(cd.x) < 0.001f ){
                        H = cd.y>0? 90.0f : 270.0f;
                    }
                    else{
                        H = atan( cd.y/cd.x )*180.0f/3.14159265f;
                    }

                    float S = cd.z*0.5f+0.5f;
                    float V = 1.0f;

                    float C = V * S;
                    float X = C * ( 1.0f - fabs( fmod( (H / 60.0f), 2.0f ) - 1.0f ) );
                    float m = V - C;

                    SCI::Vex3 col;
                    switch( (int)(H/60.0f)%6 ){
                    case 0 : col.Set( C, X, 0 ); break;
                    case 1 : col.Set( X, C, 0 ); break;
                    case 2 : col.Set( 0, C, X ); break;
                    case 3 : col.Set( 0, X, C ); break;
                    case 4 : col.Set( X, 0, C ); break;
                    case 5 : col.Set( C, 0, X ); break;
                    }
                    col += SCI::Vex3(m,m,m);



                    colormap.push_back( col.UIntColor() );
                    //colormap.push_back( (dirs[i].UnitVector()*0.5f+0.5f).UIntColor() );
                }
            }

        }

    }


    if( draw_mode == 2 ){
        if( tdata == 0 ){ return; }

        iso_points.Clear();
        iso_tris.clear();
        iso_color.clear();

        if( view_max_iso ){
            int presize = (int)iso_points.points.size();
            tdata->ExtractMaxIsosurface( *pmesh, *pdata, isoval , iso_tris, iso_points );
            iso_tris.FuseCoindidentVertices( iso_points, presize );
            iso_color.Fill( iso_tris, SCI::Vex4(0.0f,1.0f,1.0f,0.4f) );
        }
        if( view_mean_iso){
            int presize = (int)iso_points.points.size();
            tdata->ExtractMeanIsosurface( *pmesh, *pdata, isoval, iso_tris, iso_points );
            iso_tris.FuseCoindidentVertices( iso_points, presize );
            iso_color.Fill( iso_tris, SCI::Vex4(1.0f,0.0f,1.0f,0.4f) );
        }
        if( view_min_iso ){
            int presize = (int)iso_points.points.size();
            tdata->ExtractMinIsosurface( *pmesh, *pdata, isoval, iso_tris, iso_points );
            iso_tris.FuseCoindidentVertices( iso_points, presize );
            iso_color.Fill( iso_tris, SCI::Vex4(1.0f,1.0f,0.0f,0.4f) );
        }
        if( view_dim_iso ){
            int presize = (int)iso_points.points.size();
            tdata->ExtractIsosurface( *pmesh, *pdata, isoval, color_dim, iso_tris, iso_points );
            iso_tris.FuseCoindidentVertices( iso_points, presize );
            iso_color.Fill( iso_tris, SCI::Vex4(0.5f,0.5f,0.0f,0.4f) );
        }
    }

    if( draw_mode == 3 ){

        if( df_tris.size() == 0 ){
            float minv = 0, maxv = -FLT_MAX;
            Data::PointData * df_dist = new Data::PointData( 4000000, 1 );
            for(int i = -5; i < 22; i++){
                dfield->ExtractMedianIsosurface( (float)i + 0.0f, *tdata, *pmesh, df_tris, df_points, *df_dist );
                dfield->ExtractMedianIsosurface( (float)i + 0.5f, *tdata, *pmesh, df_tris, df_points, *df_dist );
            }

            for(int i = 0; i < (int)df_points.points.size(); i++){
                float val = df_dist->GetElement( i, 0 );
                if(val < 100 ){
                    minv = SCI::Min( minv, val );
                    maxv = SCI::Max( maxv, val );
                }
            }
            df_color.clear();
            df_color.SetColorPerVertex();
            for(int i = 0; i < (int)df_points.points.size(); i++){
                float val = SCI::Clamp( df_dist->GetElement( i, 0 ), minv, maxv );
                SCI::Vex4 col = SCI::lerp( SCI::Vex4(0.8f,0.8f,0.8f,0.01f), SCI::Vex4(1,0,0,0.35f), val/maxv );
                df_color.push_back( col.UIntColor() );
            }
            //df_tris.FuseCoindidentVertices( df_points );
            std::cout << minv << " " << maxv << std::endl << std::flush;
            delete df_dist;
        }

        //df_tris.SortByPainters( proj.GetMatrix() * pView->GetView() * tform, df_points, 0 );
    }

    /*
    if( draw_mode == 4 ){
        colormap.clear();
        colormap.DefaultColor() = 0xff00ffff;

        if(pdata){
            seq_cmap.LoadDefaultMapBlue();
            switch( color_mode ){
                case 0: colormap.SetByDataDimension( edge_data, seq_cmap, color_dim ); break;
                case 1: colormap.SetByDataMean( edge_data, seq_cmap );                 break;
                case 2: colormap.SetByDataStdev( edge_data, seq_cmap );                break;
            }
            colormap.SetColorPerElement();
        }
    }
    */

    if( re.parallel_coordinates ){
        /*
        if( draw_mode == 4 ){
            re.parallel_coordinates->SetData( &edge_data, &colormap );
        }
        */
        //else {
            re.parallel_coordinates->SetData( pdata, &colormap );
        //}
        re.parallel_coordinates->Reset();
    }

    if( chartCloud.parallel_coordinates ){
        /*
        if( draw_mode == 4 ){
            re.parallel_coordinates->SetData( &edge_data, &colormap );
        }
        */
        //else {
            chartCloud.parallel_coordinates->SetData( pdata, &colormap );
        //}
        chartCloud.parallel_coordinates->Reset();
    }

    chartCloud.pdata = pdata;
    chartCloud.pmesh = pmesh;
    chartCloud.tdata = tdata;
    chartCloud.NeedUpdate();
    chartCloud.draw_mode = draw_mode;
    chartCloud.iso_points = &iso_points;
    chartCloud.iso_tris = &iso_tris;
    chartCloud.iso_tets = &iso_tets;
    chartCloud.iso_hexs = &iso_hexs;
    chartCloud.iso_color = &iso_color;
    chartCloud.df_points = &df_points;
    chartCloud.df_tris = &df_tris;
    chartCloud.df_color = &df_color;
    //chartCloud.edge_data = &edge_data;
    //chartCloud.edge_mesh = &edge_mesh;
    chartCloud.cluster = &cluster;
    //chartCloud.vox_assoc = &vox_assoc;
    chartCloud.colormap = &colormap;
    chartCloud.seq_cmap = &seq_cmap;
    chartCloud.cat_cmap = &cat_cmap;
    chartCloud.draw_mode = draw_mode;
    chartCloud.color_mode = color_mode;
    chartCloud.font = &font;
    chartCloud.cluster_histogram = cluster_histogram;

    re.update();
    re.pdata = pdata;
    re.pmesh = pmesh;
    re.tdata = tdata;
    re.NeedUpdate();
    re.draw_mode = draw_mode;
    re.iso_points = &iso_points;
    re.iso_tris = &iso_tris;
    re.iso_tets = &iso_tets;
    re.iso_hexs = &iso_hexs;
    re.iso_color = &iso_color;
    re.df_points = &df_points;
    re.df_tris = &df_tris;
    re.df_color = &df_color;
    //re.edge_data = &edge_data;
    //re.edge_mesh = &edge_mesh;
    re.cluster = &cluster;
    //re.vox_assoc = &vox_assoc;
    re.colormap = &colormap;
    re.seq_cmap = &seq_cmap;
    re.cat_cmap = &cat_cmap;
    re.draw_mode = draw_mode;
    re.color_mode = color_mode;
    re.font = &font;
    re.cluster_histogram = cluster_histogram;

    re.update();

    UpdateRenderEngine2DColor( &(re2[0]) );
    UpdateRenderEngine2DColor( &(re2[1]) );
    UpdateRenderEngine2DColor( &(re2[2]) );

    pca.update();
    chartCloud.update();

}



