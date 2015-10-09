#ifndef MESHDATA_H
#define MESHDATA_H

#include <Data/VolData.h>

#include <SCI/Vex3.h>

namespace Data {

    #pragma deprecated(MeshData)

    class MeshData : public VolData {

    public:
        MeshData( const char * conn_file, const char *geom_file, const char *data_file );
        MeshData( const char * mesh_file );

        void Save( const char * fname );

        std::vector<int>       & GetConnectivity();
        std::vector<SCI::Vex3> & GetVertices();

    protected:
        std::vector<int> connectivity;
        std::vector<SCI::Vex3> vertices;
    };
}

#endif // MESHDATA_H
