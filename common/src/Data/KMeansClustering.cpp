#include <Data/KMeansClustering.h>
#include <SCI/SimpleHistogram.h>

#include <iostream>

KMeansClustering::KMeansClustering( ){
    pdata      = 0;
    iterations = 0;
    dist_type  = KM_L2Norm;
}

int       KMeansClustering::GetClusterCount() const { return clusterN; }
bool      KMeansClustering::isRunning()       const { return QThread::isRunning();  }
bool      KMeansClustering::isFinished()      const { return QThread::isFinished(); }

int KMeansClustering::GetClusterID(int id) const {
    if( id < 0 || id >= (int)clusterID.size() ) return -1;
    return clusterID[id];
}

std::vector<float> KMeansClustering::GetCluster( int id ) const {
    if( id < 0 || id >= clusterN ) return std::vector<float>();
    return cluster[id];
}

int KMeansClustering::GetClusterType() const {
    return dist_type;
}


void KMeansClustering::Initialize( const Data::PointData * _pdata, int cluster_count, int _iterations, KM_TYPE _type ){
    terminate();
    pdata = _pdata;

    std::cout << "KMeansClustering: " << "Initialization" << std::endl; fflush(stdout);
    clusterID.resize(pdata->GetVoxelCount(),0);
    cluster.resize(cluster_count);
    n_cluster.resize(cluster_count);
    n_clusterN.resize(cluster_count);

    if( _type == KM_Histogram ){
        histogram_max = pdata->GetMaximumValue(-1);
        histogram_min = pdata->GetMinimumValue(-1);
    }

    clusterN = cluster_count;

    for(int i = 0; i < clusterN; i++){
        int cl_node =  abs(rand( ) *rand() + rand()) % pdata->GetVoxelCount();
        n_cluster[i] = pdata->GetElement( cl_node );
        n_clusterN[i] = 1;
    }

    iterations = _iterations;

    dist_type = _type;

    start( QThread::LowPriority );
}


float KMeansClustering::GetDistance( int id ) const {
    std::vector<float> cur = pdata->GetElement( id );
    std::vector<float> clu = cluster[ clusterID[id] ];
    if( dist_type == KM_L2Norm ){
        return SCI::L2Norm( cur, clu );
    }
    if( dist_type == KM_Pearson ){
        return 1.0f - SCI::PearsonCorrelation( cur, clu );
    }
    if( dist_type == KM_Histogram ){
        SCI::SimpleHistogram hist0( 5, histogram_min, histogram_max );
        SCI::SimpleHistogram hist1( 5, histogram_min, histogram_max );
        hist0.AddValues(cur);
        hist1.AddValues(clu);
        float diff = 0;
        for(int i = 0; i < 5; i++){
            diff += (float)abs( hist0.GetBin(i)-hist1.GetBin(i) );
        }
        return diff;
    }
    return 0;
}

int KMeansClustering::GetClusterID( std::vector<float> cur ) const {
    int curC = -1;
    float curD = FLT_MAX;
    for(int j = 0; j < clusterN; j++){
        float d = 0;
        if( dist_type == KM_L2Norm ){
            d = SCI::L2Norm( cur, cluster[j] );
        }
        if( dist_type == KM_Pearson ){
            d = 1.0f - SCI::PearsonCorrelation( cur, cluster[j] );
        }
        if( dist_type == KM_Histogram ){
            SCI::SimpleHistogram hist0( 5, histogram_min, histogram_max );
            SCI::SimpleHistogram hist1( 5, histogram_min, histogram_max );
            hist0.AddValues(cur);
            hist1.AddValues(cluster[j]);
            d = 0;
            for(int i = 0; i < 5; i++){
                d += (float)abs( hist0.GetBin(i)-hist1.GetBin(i) );
            }
            return d;
        }

        if( d < curD ){
            curD = d;
            curC = j;
        }
    }
    return curC;
}

void KMeansClustering::run(){
    while( iterations > 0 ){
        std::cout << "KMeansClustering: " << "Iteration" << std::endl; fflush(stdout);
        for(int i = 0; i < clusterN; i++){
            cluster[i] = n_cluster[i] / (float)(n_clusterN[i]);
            n_cluster[i].clear();
            n_clusterN[i] = 0;
        }

        for(int i = 0; i < pdata->GetVoxelCount(); i++){
            std::vector<float> cur = pdata->GetElement( i );
            int curC = GetClusterID( cur );
            n_cluster[curC] += cur;
            n_clusterN[curC]++;
            clusterID[i] = curC;
        }
        iterations--;
        emit IterationComplete();
    }
    emit ClusteringComplete();

}


