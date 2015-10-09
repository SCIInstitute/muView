#ifndef TRIMESH_H
#define TRIMESH_H

#include <Data/Mesh/Primitives.h>

#include <SCI/Utility.h>
#include <SCI/Mat4.h>
#include <vector>

#include <Data/PointData.h>
#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/KDTree.h>
#include <Data/Mesh/EdgeMesh.h>
#include <Data/Mesh/ColorMesh.h>

namespace Data {

    namespace Mesh {

        class TriMesh : public std::vector<Triangle> {

        public:
            TriMesh();

            void RemoveSharedFaces( );
            //void SortByPainters(SCI::Mat4 proj, Data::Mesh::PointMesh & pnts , ColorMesh *cm);
            void SortByPainters(std::vector<int> &render_order, SCI::Mat4 proj, const PointMesh &pnts);

            void Draw(const PointMesh &pmesh, const ColorMesh & color ) const ;
            void Draw(const PointMesh &pmesh, SCI::Vex4 color) const;

            void Draw(const std::vector<int> & render_order, const PointMesh &pmesh, const ColorMesh & color ) const ;
            void Draw(const std::vector<int> & render_order, const PointMesh &pmesh, SCI::Vex4 color ) const ;

            void clear();
            //void Clear();

            SCI::Vex2 GetMinMaxArea( const PointMesh &pmesh );

            void Subdivide( float max_area, PointMesh &pmesh );

            void Append( std::vector<Triangle> & new_tris );
            void Append( std::vector<int>      & new_indices );

            void FixNormals( const PointMesh & pmesh );

            void FuseCoindidentVertices( PointMesh & pmesh, int range_min = 0, int range_max = INT_MAX, float eps = 1.0e-6 );
            //void CurvatureShading( PointMesh & pmesh );

            virtual void BuildSpacePartition( const PointMesh & verts );

            virtual void ExtractIsolineByDataDimension(EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const MultiDimensionalData & pdata_in, int dim, float iso_step );
            virtual void ExtractIsolineByMeanValue( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const MultiDimensionalData & pdata_in, float iso_step );
            virtual void ExtractIsolineByStdevValue( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const MultiDimensionalData & pdata_in, float iso_step );
            virtual void ExtractEdges( EdgeMesh & edges );
            virtual void ColorEdges( const PointMesh &pmesh, const EdgeMesh & edges, ColorMesh & cm );

        public:

            KDTree kdtree;

        };
    }
}

#endif // TRIMESH_H
