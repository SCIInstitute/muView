#include <Data/Mining/KMeansClustering.h>

#include <iostream>

float L2Norm( std::vector<float>& v0, std::vector<float>& v1){
    float sum = 0;
    for(int i = 0; i < (int)v0.size() && i < (int)v1.size(); i++){
        sum += (v0[i]-v1[i])*(v0[i]-v1[i]);
    }
    return sqrtf( sum );
}

std::vector<float> Add( std::vector<float>& v0, std::vector<float>& v1){
    std::vector<float> ret;
    int i = 0;
    while( i < (int)v0.size() && i < (int)v1.size() ){ ret.push_back( v0[i]+v1[i] ); i++; }
    while( i < (int)v0.size()                       ){ ret.push_back( v0[i] );       i++; }
    while( i < (int)v1.size()                       ){ ret.push_back( v1[i] );       i++; }
    return ret;
}

std::vector<float> Div( std::vector<float>& v0, float v1){
    std::vector<float> ret;
    for(int i = 0; i < (int)v0.size(); i++){
        ret.push_back( v0[i] / v1 );
    }
    return ret;
}

KMeansClustering::KMeansClustering( ){
    clusterCols.push_back( SCI::Vex3( 141, 211, 199 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 255, 255, 179 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 190, 186, 218 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 251, 128, 114 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 128, 177, 211 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 253, 180,  98 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 179, 222, 105 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 252, 205, 229 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 217, 217, 217 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 188, 128, 189 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 204, 235, 197 ) / 255.0f );
    clusterCols.push_back( SCI::Vex3( 255, 237, 111 ) / 255.0f );
}


SCI::Vex3 KMeansClustering::GetColor( int c ){
    return clusterCols[c%clusterCols.size()];
}

void KMeansClustering::Initialize(Data::PointData *pdata, int cluster_count ){
    std::cout << "KMeansClustering: " << "Initialization" << std::endl; fflush(stdout);
    clusterID.resize(pdata->GetVoxelCount(),0);

    clusterN = cluster_count;

    for(int i = 0; i < clusterN; i++){
        n_cluster[i] = pdata->GetElement( (rand( ) *rand() + rand()) % pdata->GetVoxelCount() );
        n_clusterN[i] = 1;
    }

}

void KMeansClustering::Iterate( Data::PointData *pdata ){
    std::cout << "KMeansClustering: " << "Iteration" << std::endl; fflush(stdout);
    for(int i = 0; i < clusterN; i++){
        cluster[i] = Div(n_cluster[i],n_clusterN[i]);
        n_cluster[i].clear();
        n_clusterN[i] = 0;
    }

    for(int i = 0; i < pdata->GetVoxelCount(); i++){
        std::vector<float> cur = pdata->GetElement( i );
        int curC = -1;
        float curD = FLT_MAX;
        for(int j = 0; j < clusterN; j++){
            float d = L2Norm( cur, cluster[j] );
            if( d < curD ){
                curD = d;
                curC = j;
            }
        }
        n_cluster[curC] = Add(n_cluster[curC],cur);
        n_clusterN[curC]++;
        clusterID[i] = curC;
    }
}



