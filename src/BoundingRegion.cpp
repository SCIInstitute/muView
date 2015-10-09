#include <BoundingRegion.h>

BoundingRegion::BoundingRegion(){ }

BBox::BBox(){
    bbmin=SCI::VEX2_MIN;
    bbmax=SCI::VEX2_MAX;
}

bool BBox::isContained( SCI::Vex2 p ){
    return ( p.x >= bbmin.x && p.x <= bbmax.x && p.y >= bbmin.y && p.y <= bbmax.y );
}

BCircle::BCircle(){
    center=SCI::Vex2(0,0);
    radius = FLT_MAX;
}

bool BCircle::isContained( SCI::Vex2 p ){
    return ( (p-center).Length() <= radius );
}

