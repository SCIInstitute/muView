#ifndef HISTOGRAM2D_H
#define HISTOGRAM2D_H

#include <SCI/Vex2.h>
#include <vector>

class Histogram2D
{
public:
    Histogram2D(int dim, int w, int h);
    ~Histogram2D();

    void Clear();

    void AddElement( SCI::Vex2 pnt, float * high_d );

    std::vector<int> GetBinList( float x, float y, float radius );
    std::vector<int> GetBinList( float x_min, float y_min, float x_max, float y_max );

    std::vector<float> GetBinMin( std::vector<int> binlist );
    std::vector<float> GetBinMax( std::vector<int> binlist );
    std::vector<float> GetBinSum( std::vector<int> binlist );
    std::vector<float> GetBinAvg( std::vector<int> binlist );
    int                GetBinCnt( std::vector<int> binlist );

protected:
    int   * bin_count;
    float * bin_sum;
    float * bin_min;
    float * bin_max;

    int w,h,dim;

};

#endif // HISTOGRAM2D_H
