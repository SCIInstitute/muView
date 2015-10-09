#ifndef FIBERDIRECTIONDATA_H
#define FIBERDIRECTIONDATA_H

#include <SCI/Vex3.h>
#include <vector>

namespace Data {
    class FiberDirectionData {

    public:
        FiberDirectionData( const char * fname, bool binary );

        void Save( const char * fname );

    public:
        std::vector<SCI::Vex3> fibs;

    };

}


#endif // FIBERDIRECTIONDATA_H
