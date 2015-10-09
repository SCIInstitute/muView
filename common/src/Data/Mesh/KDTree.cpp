#include <Data/Mesh/KDTree.h>
#include <iostream>
#include <QMutex>

using namespace Data::Mesh;


PartitionElement::PartitionElement( int idx, SCI::Vex3 cen,   float     rad   ){
    index  = idx;
    center = cen;
    radius = rad;
}

PartitionElement::PartitionElement( int idx, SCI::Vex3 bbmin, SCI::Vex3 bbmax ){
    index = idx;
    center = (bbmin+bbmax)/2.0f;
    radius = (bbmax-center).Length();
}

KDTree::KDTree( ){
    child0 = 0;
    child1 = 0;
    bb.Reset();
    elements = new std::vector<PartitionElement>();
}

KDTree::~KDTree( ){
    if( child0   ){ delete child0;   }
    if( child1   ){ delete child1;   }
    if( elements ){ delete elements; }
}

void KDTree::Clear(){
    if( child0 ){ delete child0; child0 = 0; }
    if( child1 ){ delete child1; child1 = 0; }
    if( elements ){ delete elements; elements = 0; }
    bb.Reset();
    elements = new std::vector<PartitionElement>();
}

void KDTree::InsertElement( int idx, SCI::Vex3 cen,   float     rad   ){
    elements->push_back( PartitionElement( idx, cen, rad ) );
    bb.Expand( cen-SCI::Vex3(rad,rad,rad) );
    bb.Expand( cen+SCI::Vex3(rad,rad,rad) );
}

void KDTree::InsertElement( int idx, SCI::Vex3 emin, SCI::Vex3 emax ){
    elements->push_back( PartitionElement( idx, emin, emax ) );
    bb.Expand( emin );
    bb.Expand( emax );
}

void KDTree::InsertElement( int idx, SCI::BoundingBox _bb ){
    elements->push_back( PartitionElement( idx, _bb.bbmin, _bb.bbmax ) );
    bb.Expand( _bb.bbmin );
    bb.Expand( _bb.bbmax );
}

void KDTree::InsertElement( PartitionElement pe ){
    elements->push_back( pe );
    float rad = pe.GetRadius();
    bb.Expand( pe.GetCenter()-SCI::Vex3(rad,rad,rad) );
    bb.Expand( pe.GetCenter()+SCI::Vex3(rad,rad,rad) );
}

void KDTree::Finalize( ){

    mutex.lock();

    if( elements != 0 && elements->size() > 50 ){

        SCI::Vex3 diff = bb.GetSize();

        int dim = 0;
        if( diff.x >= diff.y && diff.x >= diff.z ){
            dim = 0;
        }
        else if( diff.y >= diff.z ){
            dim = 1;
        }
        else{
            dim = 2;
        }

        float split = bb.GetCenter().data[dim];

        child0 = new KDTree( );
        child1 = new KDTree( );
        for(int i = 0; i < (int)elements->size(); i++){
            if( elements->at(i).GetCenter().data[dim] < split ){
                child0->InsertElement( elements->at(i) );
            }
            else{
                child1->InsertElement( elements->at(i) );
            }
        }
        delete elements;
        elements = 0;
    }

    mutex.unlock();

}

#include <SCI/Geometry/CompGeom.h>
void KDTree::FindPlaneIntersectionElements( SCI::Vex4 pln, std::vector<int> & elems ){
    Finalize();

    if( SCI::fabsf( PointPlaneDistance( bb.GetCenter(), pln ) ) > bb.GetRadius() ) return;

    if(child0) child0->FindPlaneIntersectionElements( pln, elems );
    if(child1) child1->FindPlaneIntersectionElements( pln, elems );

    if(elements != 0){
        for(int i = 0; i < (int)elements->size(); i++){
            float dist = SCI::fabsf( PointPlaneDistance( elements->at(i).GetCenter(), pln ) );
            if( dist <= elements->at(i).GetRadius() ){
                elems.push_back( elements->at(i).GetIndex() );
            }
        }
    }

}


void KDTree::FindClosestElements( SCI::Vex3 pnt, std::vector<int> & elems ){

    float cur_max = FLT_MAX;
    FindClosestElements_Phase1( pnt, cur_max );
    FindClosestElements_Phase2( pnt, cur_max, elems );
}

void KDTree::FindClosestElements_Phase1( SCI::Vex3 pnt, float & cur_max ){

    Finalize();

    SCI::Vex3 cen = bb.GetCenter();
    float     rad = bb.GetRadius();

    float dmin = (cen-pnt).Length() - rad;
    float dmax = dmin + rad * 2.0f;

    if( dmin > cur_max ){ return; }

    cur_max = SCI::Min( cur_max, dmax );

    if(child0) child0->FindClosestElements_Phase1( pnt, cur_max );
    if(child1) child1->FindClosestElements_Phase1( pnt, cur_max );

    if(elements != 0){
        for(int i = 0; i < (int)elements->size(); i++){
            cur_max = SCI::Min( cur_max, (pnt-elements->at(i).GetCenter()).Length() + elements->at(i).GetRadius() );
        }
    }

}

void KDTree::FindClosestElements_Phase2( SCI::Vex3 pnt, float cur_max, std::vector<int> & elems ){

    Finalize();

    SCI::Vex3 cen = bb.GetCenter();
    float     rad = bb.GetRadius();

    float dmin = (cen-pnt).Length() - rad;

    if( dmin > cur_max ){ return; }

    if(child0) child0->FindClosestElements_Phase2( pnt, cur_max, elems );
    if(child1) child1->FindClosestElements_Phase2( pnt, cur_max, elems );

    if(elements != 0){
        for(int i = 0; i < (int)elements->size(); i++){
            float dmin = (pnt-elements->at(i).GetCenter()).Length() - elements->at(i).GetRadius();
            if( dmin < cur_max ){
                elems.push_back( elements->at(i).GetIndex() );
            }
        }
    }

}

void KDTree::FindContainingElements( SCI::Vex3 pnt, std::vector<int> & elems ){
    if( pnt.x < bb.bbmin.x || pnt.y < bb.bbmin.y || pnt.z < bb.bbmin.z ) return;
    if( pnt.x > bb.bbmax.x || pnt.y > bb.bbmax.y || pnt.z > bb.bbmax.z ) return;

    Finalize();

    if(child0) child0->FindContainingElements( pnt, elems );
    if(child1) child1->FindContainingElements( pnt, elems );

    if(elements != 0){
        for(int i = 0; i < (int)elements->size(); i++){
            if( (pnt-elements->at(i).GetCenter()).Length() <= elements->at(i).GetRadius() ){
                elems.push_back( elements->at(i).GetIndex() );
            }
        }
    }
}


bool KDTree::isBuilt(){
    if( elements == 0 ){ return true; }
    if( elements->size() > 0 ){ return true; }
    return false;
}


