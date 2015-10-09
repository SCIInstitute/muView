#ifndef SHOGUN_WRAPPER_H
#define SHOGUN_WRAPPER_H

#ifndef WIN32

/*
#include <shogun/base/init.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
*/


class Shogun_Wrapper
{
public:
    static int GetMethodCount();
    static const char * GetMethodName(int id);

    Shogun_Wrapper();
    ~Shogun_Wrapper();

    void SetMethod( int method );
    void SetSize( int _elemN, int _low_dim, int _high_dim );

    void Run( );

    void SetInputData( int elem, float  * vals );
    void SetInputData( int elem, double * vals );

    void GetOutputData( int elem, double * vals ) const ;
    void GetOutputData( int elem, float  * vals ) const ;

    void GetMappedPoint( double * highd_in, double * lowd_out ) const ;

protected:
    double *  input_matrix;
    double * output_matrix;

    int high_dim;
    int low_dim;
    int elemN;

    void * _preprocessor;
    void * _kernel;

    int method;

    //shogun::SGMatrix<double> inmat;

/*



    void * redmet;
*/


};

#endif

#endif // SHOGUN_WRAPPER_H
