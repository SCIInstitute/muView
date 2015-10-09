#ifndef RENDERENGINE_H
#define RENDERENGINE_H


//#include <Data/VolData.h>
//#include <Data/MeshData.h>
#include <Data/VolumetricData.h>
#include <Data/PointData.h>
#include <Data/MappedData.h>
#include <Data/Mesh/TetMesh.h>
#include <Data/Mesh/HexMesh.h>
#include <Data/Mesh/SolidMesh.h>
#include <Data/Mesh/EdgeMesh.h>
#include <Data/FiberDirectionData.h>
#include <Data/KMeansClustering.h>
#include <Data/DistanceField.h>
#include <Data/Mesh/OrientationMesh.h>

#include <SCI/Timer.h>
#include <SCI/Subset.h>
#include <SCI/Vex3.h>
#include <SCI/Colormap.h>

#include <GL/oglFont.h>

#include <muView/Histogram.h>
#include <muView/ParallelCoordinates.h>
#include <muView/RenderEngine3D.h>
#include <muView/RenderEngine2D.h>
#include <muView/PCAView.h>

class RenderEngine : public QObject {

    Q_OBJECT

public:
    RenderEngine( SCI::ThirdPersonCameraControls * _pView );

    void SetFiberData( Data::FiberDirectionData * fdata );
    void SetData( Data::PointData * pdata, Data::Mesh::PointMesh *pmesh, Data::Mesh::SolidMesh *tdata );
    void SetDistanceFieldData( Data::DistanceFieldSet * _dfield );
    void AddImportedMesh( Data::Mesh::PointMesh *pmesh, Data::Mesh::SolidMesh *tdata );

public slots:

    void setDrawModePoints( );
    void setDrawModeVolumeRendering( );
    void setDrawModeIsosurfacing( );
    void setDrawModeDistanceField( );
    void setDrawModeNetwork( );

    void setColorModeDimension( );
    void setColorModeMin( );
    void setColorModeMedian( );
    void setColorModeMax( );
    void setColorModeStDev( );
    void setColorModeCluster( );
    void setColorModeIsovalue( );
    void setColorModePCA();
    void setColorModeFibers();

    void setDimension( int );
    void setClusterCount( int );
    void setClusterIterations( int );
    void setClusterRecalculate( );
    void setClusterHistogram( bool );
    void setClusterTypeL2Norm( );
    void setClusterTypePearson( );
    void setClusterTypeHistogram( );

    void setFiberDirection( bool );
    void setFiberLength( double );

    void setIsovalue( double );
    void setDimIsosurface( bool );
    void setMinIsosurface( bool );
    void setMeanIsosurface( bool );
    void setMaxIsosurface( bool );

    void setClipXVal( double );
    void setClipYVal( double );
    void setClipZVal( double );

    void setClipXFlip( );
    void setClipYFlip( );
    void setClipZFlip( );

    void setClipXEnable( bool );
    void setClipYEnable( bool );
    void setClipZEnable( bool );

    void calculateSubVolume();

    void UpdateRenderEngine2DColor(RenderEngine2D * re );

    void Recalculate(  );


protected:
    int  draw_mode;
    int  color_mode;
    int  color_dim;
    int  clusterN;
    int  clusterI;
    bool cluster_histogram;
    KMeansClustering::KM_TYPE cluster_type;



    float isoval;
    bool view_dim_iso;
    bool view_min_iso;
    bool view_mean_iso;
    bool view_max_iso;
    //bool view_iso_tet;

    bool view_fibers;
    float fiber_length;

    KMeansClustering cluster;

    SCI::SequentialColormap  seq_cmap;
    SCI::CatagoricalColormap cat_cmap;

    // Containers for data
    Data::PointData          * pdata;
    Data::Mesh::PointMesh    * pmesh;
    Data::Mesh::SolidMesh    * tdata;
    Data::FiberDirectionData * fdata;
    Data::DistanceFieldSet   * dfield;
    Data::Mesh::ColorMesh      colormap;

    //Data::DenseMultiDimensionalData edge_data;
    //Data::Mesh::EdgeMesh            edge_mesh;


    Data::Mesh::PointMesh iso_points;
    Data::Mesh::TriMesh   iso_tris;
    Data::Mesh::TetMesh   iso_tets;
    Data::Mesh::HexMesh   iso_hexs;
    Data::Mesh::ColorMesh iso_color;

    Data::Mesh::PointMesh df_points;
    Data::Mesh::TriMesh   df_tris;
    Data::Mesh::ColorMesh df_color;

    //VoxelAssociativity vox_assoc;

    oglWidgets::oglFont font;


public:

    RenderEngine3D re;
    RenderEngine2D re2[3];
    PCAView        pca;

    void SetParallelCoordinateView( ParallelCoordinates * pc ){ re.parallel_coordinates = pc; }

    Data::Mesh::ColorMesh * GetColorMesh(){ return &colormap; }


};

#endif // RENDERENGINE_H
