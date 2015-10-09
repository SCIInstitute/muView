#ifndef NRRDDATA_H
#define NRRDDATA_H

#include <Data/VolData.h>
#include <volume.h>

namespace Data {

    // Nrrd specific data container
    class NrrdData : public VolData {
    public:
        NrrdData( NrrdVolume<float> * data );

    };
}

#endif // NRRDDATA_H
