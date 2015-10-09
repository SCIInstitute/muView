#ifndef BARCHART_H
#define BARCHART_H

#include <vector>

#include <Drawable/DrawableObject.h>

namespace Drawable {

    class BarChart : public DrawableObject {

    public:
        BarChart();

        virtual void Draw( );

        void SetMinimumValue( float v );
        void SetMaximumValue( float v );

        void SetAverage( std::vector<float> vals );
        void SetMinimum( std::vector<float> vals );
        void SetMaximum( std::vector<float> vals );

    protected:
        std::vector<float> maxval;
        std::vector<float> minval;
        std::vector<float> avg;

        float vmin, vmax;

    };

}

#endif // BARCHART_H
