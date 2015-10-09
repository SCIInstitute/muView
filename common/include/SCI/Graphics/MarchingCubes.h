#ifndef __SCI_GRAPHICS_MARCHINGCUBES__
#define __SCI_GRAPHICS_MARCHINGCUBES__

#include <SCI/Vex3.h>

#include <vector>

namespace SCI {
    namespace Graphics {

        class MarchingCubes {
        public:
            MarchingCubes( );
            virtual ~MarchingCubes();

            void ProcessTriangles( int index_base, std::vector<int> & index_out, float vals[2][2][2], float iso_val );
            void ProcessVertices(  std::vector<SCI::Vex3>            & verts_out, SCI::Vex3          vertices[2][2][2], float vals[2][2][2], float iso_val);
            void ProcessParameters(std::vector< std::vector<float> > & param_out, std::vector<float> param_in[2][2][2], float vals[2][2][2], float iso_val);

        private:

            short get_lookup(float vals[2][2][2], float iso_val );

            const static int edgeTable[256];
            const static int triTable[256][16];

            SCI::Vex3 interpolate( SCI::Vex3 vert0, SCI::Vex3 vert1, float val0, float val1, float iso_val );
            std::vector<float> interpolate( std::vector<float> & vert0, std::vector<float> & vert1, float val0, float val1, float iso_val );

        };
    }
}

#endif // __SCI_GRAPHICS_MARCHINGCUBES__
