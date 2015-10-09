#ifndef KNEARESTNEIGHBOR_H
#define KNEARESTNEIGHBOR_H

#include <Data/PhysicsData.h>

class KNearestNeighbor
{
public:
    KNearestNeighbor();

    void SetData( Data::PhysicsData & input, Data::PhysicsData & output );

    std::vector<float> Find( std::vector<float> point, int k );

protected:
    Data::PhysicsData * input;
    Data::PhysicsData * output;

};

#endif // KNEARESTNEIGHBOR_H
