#ifndef BOUNDINGREGION_H
#define BOUNDINGREGION_H

#include <SCI/Vex2.h>

class BoundingRegion {

public:
    BoundingRegion();

    virtual bool isContained( SCI::Vex2 p ) = 0;
};


class BBox : public BoundingRegion {

public:
    SCI::Vex2 bbmin;
    SCI::Vex2 bbmax;

public:
    BBox();

    virtual bool isContained( SCI::Vex2 p );
};

class BCircle : public BoundingRegion {

public:
    SCI::Vex2 center;
    float radius;

public:
    BCircle();

    virtual bool isContained( SCI::Vex2 p );
};


#endif // BOUNDINGREGION_H
