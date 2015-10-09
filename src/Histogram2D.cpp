#include <Histogram2D.h>

Histogram2D::Histogram2D(int _dim, int _w, int _h) : dim(_dim), w(_w), h(_h) {
    bin_count = 0;
    bin_sum = bin_min = bin_max = 0;

    bin_count = new int  [w*h];
    if(dim < 2500){
        bin_sum   = new float[w*h*dim];
        bin_min   = new float[w*h*dim];
        bin_max   = new float[w*h*dim];
    }
}

Histogram2D::~Histogram2D(){
    if(bin_count) delete [] bin_count;
    if(bin_sum)   delete [] bin_sum;
    if(bin_min)   delete [] bin_min;
    if(bin_max)   delete [] bin_max;
}

void Histogram2D::Clear(){
    for(int i = 0; i < w*h; i++){
        bin_count[i] = 0;
        for(int j = 0; j < dim; j++){
            if(bin_sum) bin_sum[i*dim+j] = 0;
            if(bin_min) bin_min[i*dim+j] = 1;
            if(bin_max) bin_max[i*dim+j] = 0;
        }
    }
}

void Histogram2D::AddElement( SCI::Vex2 pnt, float *high_d ){
    int x = (int)((pnt.x+1.0f)*((float)(w-1))/2.0f);
    int y = (int)((pnt.y+1.0f)*((float)(h-1))/2.0f);

    if( x >= 0 && x < w && y >= 0 && y < h ){
        int loc = y*w+x;
        for(int i = 0; i < dim; i++){
            if(bin_sum) bin_sum[loc*dim+i] += high_d[i];
            if(bin_min) bin_min[loc*dim+i] = SCI::Min( bin_min[loc*dim+i], high_d[i] );
            if(bin_max) bin_max[loc*dim+i] = SCI::Max( bin_max[loc*dim+i], high_d[i] );
        }
        bin_count[loc]++;
    }
}

std::vector<int> Histogram2D::GetBinList( float xf, float yf, float radius ){
    std::vector<int> ret;
    if( radius < 0 ){
        int x = (int)((xf+1.0f)*((float)w-1)/2.0f);
        int y = (int)((yf+1.0f)*((float)h-1)/2.0f);
        if( x >= 0 && x < w && y >= 0 && y < h ){
            ret.push_back( y*w+x );
        }
    }
    else{
        int xmin = (int)((xf-radius+1.0f)*((float)w-1)/2.0f);
        int xmax = (int)((xf+radius+1.0f)*((float)w-1)/2.0f);
        int ymin = (int)((yf-radius+1.0f)*((float)h-1)/2.0f);
        int ymax = (int)((yf+radius+1.0f)*((float)h-1)/2.0f);

        for(int y = SCI::Max(0,ymin); y <= SCI::Min(h-1,ymax); y++){
            for(int x = SCI::Max(0,xmin); x <= SCI::Min(w-1,xmax); x++){
                float cx = (float)x/((float)w-1)*2.0f-1.0f;
                float cy = (float)y/((float)h-1)*2.0f-1.0f;
                if( (powf(cx-xf,2.0f)+powf(cy-yf,2.0f)) <= powf(radius,2.0f) ){
                    ret.push_back( y*w+x );
                }
            }
        }
    }
    return ret;
}

std::vector<int> Histogram2D::GetBinList( float x_min, float y_min, float x_max, float y_max ){
    std::vector<int> ret;

    int xmin = (int)((x_min+1.0f)*((float)w-1)/2.0f);
    int xmax = (int)((x_max+1.0f)*((float)w-1)/2.0f);
    int ymin = (int)((y_min+1.0f)*((float)h-1)/2.0f);
    int ymax = (int)((y_max+1.0f)*((float)h-1)/2.0f);

    for(int y = SCI::Max(0,ymin); y <= SCI::Min(h-1,ymax); y++){
        for(int x = SCI::Max(0,xmin); x <= SCI::Min(w-1,xmax); x++){
            ret.push_back( y*w+x );
        }
    }
    return ret;
}

std::vector<float> Histogram2D::GetBinMin( std::vector<int> poslist ){
    //int dim = data_in->GetDim();
    std::vector<float> ret(dim,FLT_MAX);
    if(bin_min){
        for(int i = 0; i < (int)poslist.size(); i++){
            for(int j = 0; j < dim; j++){
                ret[j] = SCI::Min(ret[j],bin_min[poslist[i]*dim+j]);
            }
        }
    }
    return ret;
}

std::vector<float> Histogram2D::GetBinMax( std::vector<int> poslist ){
    //int dim = data_in->GetDim();
    std::vector<float> ret(dim,-FLT_MAX);
    if(bin_max){
        for(int i = 0; i < (int)poslist.size(); i++){
            for(int j = 0; j < dim; j++){
                ret[j] = SCI::Max(ret[j],bin_max[poslist[i]*dim+j]);
            }
        }
    }
    return ret;
}

std::vector<float> Histogram2D::GetBinSum( std::vector<int> poslist ){
    //int dim = data_in->GetDim();
    std::vector<float> ret(dim,0);
    if(bin_sum){
        for(int i = 0; i < (int)poslist.size(); i++){
            for(int j = 0; j < dim; j++){
                ret[j] += bin_sum[poslist[i]*dim+j];
            }
        }
    }
    return ret;
}

int Histogram2D::GetBinCnt( std::vector<int> poslist ){
    int ret = 0;
    if(bin_count){
        for(int i = 0; i < (int)poslist.size(); i++){
            ret += bin_count[poslist[i]];
        }
    }
    return ret;
}


std::vector<float> Histogram2D::GetBinAvg( std::vector<int> poslist ){
    //int dim = data_in->GetDim();
    std::vector<float> ret(dim,0);
    if(bin_sum && bin_count){
        int sum = 0;
        for(int i = 0; i < (int)poslist.size(); i++){
            for(int j = 0; j < dim; j++){
                ret[j] += bin_sum[poslist[i]*dim+j];
            }
            sum += bin_count[poslist[i]];
        }
        for(int j = 0; j < dim; j++){
            ret[j] = ret[j]/(float)sum;
        }
    }
    return ret;
}
