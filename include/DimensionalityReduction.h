#ifndef DIMENSIONALITYREDUCTION_H
#define DIMENSIONALITYREDUCTION_H

#include <QObject>
//#include <Data/BasicData.h>
//#include <Data/ProxyData.h>
#include <Data/MultiDimensionalData.h>
#include <SCI/Subset.h>

#include <QThread>
#include <DRL_Wrapper.h>
#include <Shogun_Wrapper.h>

#ifdef WIN32
    #define DR_WRAPPER DRL_Wrapper
#else
    #define DR_WRAPPER Shogun_Wrapper
#endif

class DimensionalityReduction {

public:
    static int          GetMethodCount();
    static const char * GetMethodName( int i );

public:
    DimensionalityReduction();
    ~DimensionalityReduction();

    // Start a dimensionality reduction. Function will block until the feature points identify a map which can be used on the elements. New thread will be launched to reduce the elements.
    void Start( int method, Data::MultiDimensionalData & _data_in, SCI::Subset & _features, Data::MultiDimensionalData & _data_out, SCI::Subset & _elements );
    void StartAxis( int method, Data::MultiDimensionalData & _data_in, SCI::Subset & _features, Data::MultiDimensionalData & _data_out, SCI::Subset & _elements );
    void Restart( SCI::Subset & _elements );
    void Stop( );


    double* GetPrincipalComponent(int component);
    // Return the total number of feature points
    int GetFeatureCount() const ;
    // Gets the reduced location of a feature point
    void GetFeaturePoint( int elem, float  * pout ) const ;
    void GetFeaturePoint( int elem, double * pout ) const ;
    // Gets the Voxel ID of a Feature
    int  GetFeatureVoxID( int elem ) const ;

    // Return the total number of elements which have been mapped to lower dimension
    int GetElementCount() ;
    // Return the total number of elements
    int GetTotalElementCount() const ;
    // Gets the reduced location of an element point
    void GetElementPoint( int elem, float  * pout ) const ;
    void GetElementPoint( int elem, double * pout ) const ;
    // Gets the Voxel ID of an Element
    int  GetElementVoxID( int elem ) const ;

    // Maps a point (pin) from high dimensional space to low dimensional space
    void GetMappedPoint( float  * pin, float  * pout ) const ;
    void GetMappedPoint( double * pin, double * pout ) const ;

    // Return the number of input (high) dimensions
    int GetHighDimension() const ;
    // Return the number of output (low) dimensions
    int GetLowDimension() const ;

    // Functions for getting the bounding box of the low dimensional data
    float GetMinValue( int dim ) const ;
    float GetMaxValue( int dim ) const ;

protected:

    int high_dimN;
    int low_dimN;

    float * min_elem;
    float * max_elem;

    DR_WRAPPER dr;

    void processNextElement();

public:

    int cur_method;

    SCI::Subset * features;
    SCI::Subset * elements;

    Data::MultiDimensionalData * data_in;
    Data::MultiDimensionalData * data_out;

    int e_comp;

};


#endif // DIMENSIONALITYREDUCTION_H
