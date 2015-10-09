#ifndef MULTIDIMENSIONALDATAVECTOR_H
#define MULTIDIMENSIONALDATAVECTOR_H

#include <Data/MultiDimensionalData.h>

namespace Data {

    // Base class for volume holding data containers
    class MultiDimensionalDataVector : public MultiDimensionalData {
    public:
        MultiDimensionalDataVector( int _elemN, int _dimN );

        // Get a rough estimate of the size of the data contained in the class
        virtual int GetDataSize() const ;

        virtual void Resize( int _elemN, int _dimN );

        // Various functions for setting values
        virtual void SetElement( int elem_id, const std::vector<float> & val );
        virtual void SetElement( int elem_id, int dim, float val );
        virtual void SetElement( int elem_id, const float  * val );
        virtual void SetElement( int elem_id, const double * val );

        virtual void PushElement( const std::vector<float> & val );
        virtual void PushElement( const float  * val );
        virtual void PushElement( const double * val );


        // Various functions for getting values
        virtual SCI::VexN GetElement( int elem_id )                 const ;
        virtual float     GetElement( int elem_id, int dim )        const ;
        virtual void      GetElement( int elem_id, float  * space ) const ;
        virtual void      GetElement( int elem_id, double * space ) const ;

        virtual float              GetMaximumValue( int dim = -1 ) ;
        virtual float              GetMinimumValue( int dim = -1 ) ;

    protected:
        std::vector<float> data;

        void RecalculateMinumumAndMaximum();


    };



}


#endif // MULTIDIMENSIONALDATAVECTOR_H
