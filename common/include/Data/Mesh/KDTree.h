#ifndef KDTREE_H
#define KDTREE_H

#include <SCI/Vex3.h>
#include <SCI/Vex4.h>
#include <SCI/BoundingBox.h>

#include <vector>
#include <QMutex>

namespace Data {

    namespace Mesh {

        class PartitionElement {
        public:

            PartitionElement( int idx, SCI::Vex3 cen,   float     rad   );
            PartitionElement( int idx, SCI::Vex3 bbmin, SCI::Vex3 bbmax );

            float     GetRadius(){ return radius; }
            SCI::Vex3 GetCenter(){ return center; }
            int       GetIndex(){  return index;  }

        protected:
            SCI::Vex3 center;
            float     radius;
            int       index;

        };

        class KDTree {

        public:
            KDTree( );
            ~KDTree( );

            void Clear();

            void InsertElement( int idx, SCI::Vex3 cen,   float     rad   );
            void InsertElement( int idx, SCI::Vex3 emin, SCI::Vex3 emax );
            void InsertElement( int idx, SCI::BoundingBox bb );
            void InsertElement( PartitionElement pe );

            void FindPlaneIntersectionElements( SCI::Vex4 pln, std::vector<int> & elems );
            void FindContainingElements( SCI::Vex3 pnt, std::vector<int> & elems );
            void FindClosestElements( SCI::Vex3 pnt, std::vector<int> & elems );

            bool isBuilt();

        public:
            void FindClosestElements_Phase1( SCI::Vex3 pnt, float &cur_max );
            void FindClosestElements_Phase2( SCI::Vex3 pnt, float  cur_max, std::vector<int> & elems );

            void Finalize( );

            std::vector<PartitionElement> * elements;

            //SCI::Vex3 bbmin, bbmax;
            SCI::BoundingBox bb;

            KDTree * child0;
            KDTree * child1;

            QMutex mutex;

        };
    }
}



#endif // KDTREE_H
