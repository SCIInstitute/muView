// = = = = = = = = = = = = = = = = //
// Calculate the entropy volume    //
// Kristi Potter 2011              //
// = = = = = = = = = = = = = = = = //
#include <iostream>
#include <volume.h>

int main( int argc, char * argv[] ) 
{
  std::string infile = "/Users/kpotter/Code/Entropy/data/lesion/fuzzy/lesionFuzzyALL.nrrd";
  std::string outfile = "/Users/kpotter/Code/Entropy/data/lesion/fuzzy/lesionFuzzyEntropyLog2.nrrd";
  
  // Read in a data file
  NrrdVolume<float> * vol = new NrrdVolume<float>(infile);
  vol->printDims();

  std::cout << "Calculate Entropy!" << std::endl;  

  // Calculate the entropy volume
  EntropyVolume<float> *entropyVol = new EntropyVolume<float>(vol);
  entropyVol->writeNrrd(outfile);
}

