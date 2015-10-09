#include <Data/Mesh/OrientationMesh.h>

using namespace Data::Mesh;

OrientationMesh::OrientationMesh(){ }

void OrientationMesh::Calculate( TriMesh & tmesh, PointMesh & pmesh ){
    resize( pmesh.points.size(), SCI::Vex3(0,0,0) );
    for(int i = 0; i < (int)tmesh.size(); i++){
        Triangle  tri  = tmesh[i];
        SCI::Vex3 norm = tri.GetNormal( pmesh );
        for(int j = 0; j < 3; j++){
            int idx = tri.data[j];
            float d0 = fabsf( SCI::dot( at(idx),  norm ) );
            float d1 = fabsf( SCI::dot( at(idx), -norm ) );
            at(idx) += ( d0 < d1 ) ? norm : -norm;
        }
    }
    for(int idx = 0; idx < (int)size(); idx++){
        SCI::Vex3 n = at(idx).UnitVector();
        int mdim = 0;
        if( fabsf(n.x) > fabsf(n.y) && fabsf(n.x) > fabsf(n.z) ) mdim = 0;
        if( fabsf(n.y) > fabsf(n.x) && fabsf(n.y) > fabsf(n.z) ) mdim = 1;
        if( fabsf(n.z) > fabsf(n.x) && fabsf(n.z) > fabsf(n.y) ) mdim = 2;
        if( n.data[mdim] < 0 ){ n *= -1.0f; }
        at(idx) = n;
    }
}

void OrientationMesh::ZeroFlatRegions( TriMesh & tmesh, PointMesh & pmesh ){
    std::vector< float > diff( size(), 0 );
    for(int i = 0; i < (int)tmesh.size(); i++){
        Triangle  tri  = tmesh[i];
        SCI::Vex3 norm = tri.GetNormal( pmesh );
        for(int j = 0; j < 3; j++){
            int idx = tri.data[j];
            diff[idx] += 1.0f - fabsf( SCI::dot( at(idx),  norm ) );
        }
    }
    for(int idx = 0; idx < (int)size(); idx++){
        if( diff[idx] < 0.00001f ){
            at(idx) = SCI::Vex3(0,0,0);
        }
    }
}

void OrientationMesh::Smooth(const TriMesh & tmesh , int iterations){
    for(int iter = 0; iter < iterations; iter++){
        std::vector<SCI::Vex3> new_list ( size(), SCI::Vex3(0,0,0) );
        for(int i = 0; i < (int)tmesh.size(); i++){
            Triangle  tri  = tmesh[i];
            for(int j = 0; j < 3; j++){
                int idx0 = tri.data[j];
                for(int k = 0; k < 3; k++){
                    int idx1 = tri.data[k];
                    float d0 = fabsf( SCI::dot( new_list[idx0],  at(idx1) ) );
                    float d1 = fabsf( SCI::dot( new_list[idx0], -at(idx1) ) );
                    new_list[idx0] += ( d0 < d1 ) ? at(idx1) : -at(idx1);
                }
            }
        }

        for(int i = 0; i < (int)new_list.size(); i++){
            at(i) = new_list[i].UnitVector();
        }
    }
}

