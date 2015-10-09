#ifndef KMEANSCLUSTERING_H
#define KMEANSCLUSTERING_H

#include <QThread>

#include <vector>

#include <SCI/Vex3.h>
#include <SCI/VexN.h>

#include <Data/PointData.h>
#include <SCI/Colormap.h>

class KMeansClustering : public QThread {

    Q_OBJECT


public:
    enum KM_TYPE {
        KM_L2Norm = 0,
        KM_Pearson = 1,
        KM_Histogram = 2
    };

    KMeansClustering( );

    void Initialize( const Data::PointData * _pdata, int cluster_count, int _iterations, KM_TYPE type );

    int GetClusterCount() const ;
    int GetClusterID(int id) const ;
    int GetClusterID( std::vector<float> pnt ) const ;
    std::vector<float> GetCluster( int id ) const ;
    int GetClusterType() const ;

    float GetDistance( int id ) const ;

    bool isRunning() const ;
    bool isFinished() const ;

signals:
    void IterationComplete();
    void ClusteringComplete();

protected:

    virtual void run();

protected:
    const Data::PointData  * pdata;

    std::vector< int >       clusterID;
    std::vector< SCI::VexN > cluster;
    std::vector< SCI::VexN > n_cluster;
    std::vector< int >       n_clusterN;
    int                      clusterN;
    int                      iterations;
    KM_TYPE                  dist_type;

    float histogram_min, histogram_max;

};

#endif // KMEANSCLUSTERING_H
