#ifndef KMEANSCLUSTERING_H
#define KMEANSCLUSTERING_H

#include <vector>

#include <SCI/Vex3.h>

#include <Data/PointData.h>

class KMeansClustering {

public:
    KMeansClustering( );

    void Initialize( Data::PointData * pdata, int cluster_count );
    void Iterate(Data::PointData *pdata );

    SCI::Vex3 GetColor( int c );

protected:
    std::vector< SCI::Vex3 > clusterCols;

public:

    std::vector< int > clusterID;
    std::vector<float> cluster[20];
    std::vector<float> n_cluster[20];
    int                n_clusterN[20];
    int                clusterN;

};

#endif // KMEANSCLUSTERING_H
