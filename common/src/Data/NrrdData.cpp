#include <Data/NrrdData.h>

using namespace Data;

// NrrdData is just a wrapper for VolData class
NrrdData::NrrdData( NrrdVolume<float> * data ) : VolData( data->getDim(0), data->getDim(1), data->getDim(2), data->getNumChannels(), data->getData() ){ }

