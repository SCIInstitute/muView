// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Volume classes hold the data
// Kristi Potter 2011
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifndef _VOLUME_H
#define _VOLUME_H

//#define TEST
#define TEEM_STATIC
#include <teem/nrrd.h>
#include <iostream>

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~//
// Volume class to access the data
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~//
template <class T> class Volume{
 public:
  // --- Constructor --- //
  Volume() {}
  
  // --- Destructor --- //
  virtual ~Volume(){}

  // --- Return the requested dimension --- //
  int getDim(int i) { if (i >= 0 && i < 4) return _dims[i]; else return -1; }

  // --- Return the array of dimensions --- //
  int * getDims() { return _dims; }

  // --- Return the total number of voxels (x*y*z) --- //
  int getNumVoxels() { return _dims[0]*_dims[1]*_dims[2]; }

  // --- Return the number of channels (# of volumes) --- //
  int getNumChannels() { return _dims[3]; }

  // --- Get the data value at a voxel --- //
  T getValue(int x, int y, int z, int c){ return _data[index(x,y,z,c)];}

  // --- Set the data value at a voxel --- //
  void setValue(T value, int x, int y, int z, int c){ _data[index(x,y,z,c)] = value; }

  // --- Set the data --- //
  void setData(T * data, int dim0, int dim1, int dim2, int dim3)
  { _data=data; _dims[0]=dim0; _dims[1]=dim1; _dims[2]=dim2; _dims[3]=dim3;}

  // --- Print the dimensions --- //
  void printDims() { std::cout << "Dims: "<<_dims[0]<<" x "<< _dims[1]<<" x "<<_dims[2]<<" x "<<_dims[3]<<std::endl;}

 protected:

  // The 4D dimension of the data volume
  int _dims[4];

  // The data is a 4D volume
  T * _data;

  //--- Get the 4D index into the volume --- //
  inline const int index(int x, int y, int z, int c) 
  { return ((c*_dims[2]*_dims[1]*_dims[0]) + (z*_dims[0]*_dims[1]) + (y*_dims[0]) + x); }
  
};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~//
// A nrrd volume
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~//  
template <class T> 
class NrrdVolume : public Volume<T>{
 public:
  
  enum TypeEnum{ FLOAT=0, DOUBLE };

  // --- Contructors --- //
  NrrdVolume(){ _dims[0]=_dims[1]=_dims[2]=_dims[3]=0; }

  NrrdVolume(std::string filename) 
  { 
    std::cout << "Reading nrrd" << std::flush;

    Nrrd * nrrd = nrrdNew();
    //    NrrdIoState *nio = nrrdIoStateNew();
    FILE *file = fopen(filename.c_str(), "rb" );
    // nio->skipData = 1;
    if( nrrdLoad(nrrd, filename.c_str(), NULL) )
      {
	std::cout << "Load Error: " << filename <<std::endl;
    //exit(1);
        _data = 0; return;
      }
    fclose(file);
    
    // Find the dimensions
    _dims[0] = nrrd->axis[0].size;
    _dims[1] = nrrd->axis[1].size;
    _dims[2] = nrrd->axis[2].size;
    if(nrrd->dim == 4)
      _dims[3] = nrrd->axis[3].size;

    // Set the data
    _data =(T *)(nrrd->data);

    std::cout << " done." << std::endl; 
  }
  
  // --- Destructor --- //
  virtual ~NrrdVolume() {}
 
  // --- Save the nrrd --- //
  void writeNrrd(std::string filename = "out.nrrd", int typeEnum = FLOAT){
    Nrrd *nrrd = nrrdNew(); 
    size_t  dimArray[] = {_dims[0], _dims[1], _dims[2]};
    if(typeEnum == FLOAT)
      nrrdWrap_nva(nrrd, _data, nrrdTypeFloat, 3, dimArray); 
    else if(typeEnum == DOUBLE)
      nrrdWrap_va(nrrd, _data, nrrdTypeDouble, 3, dimArray);
   
    nrrdSave(filename.c_str(), nrrd, NULL);
    nrrdNix(nrrd);
  }

  bool Loaded(){ return _data != 0; }

  T * getData(){ return _data; }

 protected:

  using Volume<T>::_dims;
  using Volume<T>::_data;
};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~//
// A nrrd entropy volume
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~//
template<class T>
class EntropyVolume : public NrrdVolume<T>{
  
public:
  // --- Constructor calculates entropy --- //
  EntropyVolume(Volume<T> *volume) : NrrdVolume<T>() {

    // Set the dimsions of the entropy volume
    _dims[0] = volume->getDim(0); _dims[1] = volume->getDim(1); _dims[2] = volume->getDim(2);
    _data = new T[_dims[0]*_dims[1]*_dims[2]];
    

    int startX = 0; int startY = 0; int startZ = 0;
    int endX = volume->getDim(0);
    int endY = volume->getDim(1);
    int endZ = volume->getDim(2);

#ifdef TEST
    startX = 91;  endX=92;
    startY = 109; endY = 110;
    startZ = 0;   endZ=1;
#endif
    
    for(int z = startZ; z < endZ; z++)
      for(int y = startY; y < endY; y++)
	for(int x = startX; x < endX; x++)
 	{
	  T ent = 0;
	  
#ifdef TEST
	  T test = 0;
	  std::cout << "P: " << std::endl;
#endif
 	  // Go through each channel
 	  for(int c = 0; c < volume->getDim(3); c++)
 	    {
	      T p = volume->getValue(x,y,z,c);
#ifdef TEST
	      std::cout << p << " ";
	      test += p;
#endif
	      
	      if(p != 0)
		ent -= p*log2(p);
 	    }
#ifdef TEST
	  std::cout << std::endl;
	  std::cout << "test: " << test << std::endl;
	  std::cout << "ent: " << ent << std::endl;
#endif
	  this->setValue(ent,x,y,z,0);
 	}
  }

protected:
  using Volume<T>::_dims;
  using Volume<T>::_data;

};

  
#endif
