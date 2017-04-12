#include <QtGui>

#include "muView/ChartRect.h"

ChartRect::ChartRect(QPointF &position, float *data, int numData, int w, int h, float min, float max, int steps)
    : position(position), minGlobal(min), maxGlobal(max), steps(steps){

    for (int i=0; i<numData;i++){
        this->data.push_back(data[i]);
    }


    for (int i=0; i<steps;i++){
        axisTicks.push_back(min + i*(max-min)/steps);
    }


    QChart *chart = new QChart();
    chart->setMargins(QMargins(1,1,1,1));//(int left, int top, int right, int bottom)




    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);



    resize(w,h);

    computeBinning(minGlobal, maxGlobal, steps);
}

QPixmap ChartRect::grabChartView(){
    return this->chartView->grab();
}


void ChartRect::setData(float *data, int numData){

    chartView->chart()->removeAllSeries();

    /*
    QLineSeries *series = new QLineSeries();
    for (int i=0; i<numData;i++){
        this->data.push_back(data[i]);
        *series << QPointF(i, data[i]);
    }

    //compute distribution



    chartView->chart()->addSeries(series);
    */
    this->data.clear();

    for (int i=0; i<numData;i++){
        this->data.push_back(data[i]);
    }


    computeBinning(minGlobal, maxGlobal, steps);
}

void ChartRect::computeBinning(float min, float max, int steps){



    // data is in between bin and bin+1
    std::vector<int> binCounts;
    for (int i=0;i<axisTicks.size();i++){
        binCounts.push_back(0);
    }

    for (int i=0;i<data.size();i++){
        int a = (data[i] - axisTicks[0])/((max-min)/steps);
        binCounts[a] +=1;
    }

    QLineSeries *series = new QLineSeries();

    for (int i=0;i<axisTicks.size();i++){
        *series << QPointF(axisTicks[i], binCounts[i]);
    }

    chartView->chart()->addSeries(series);
    chartView->chart()->legend()->hide();
    chartView->chart()->createDefaultAxes();
}

void ChartRect::setLocation(SCI::Vex3 location){
    this->location = location;
}

void ChartRect::setPosition(QPointF pos){
    this->position = pos;
}

void ChartRect::resize(int w, int h){
    this->chartView->resize(w, h);
    this->w = w;
    this->h = h;
}

float ChartRect::width(){
    return this->chartView->size().width();
}

float ChartRect::height(){
    return this->chartView->size().height();
}


void ChartRect::drawChartRect(QPainter *painter, QPixmap pix)
{
    painter->save();
    painter->drawPixmap(position.x(), position.y(), w, h, pix);
    QImage * img = new QImage();
    *img = QImage(w*50,h*50, QImage::Format_ARGB32);
    //QString path = "/Users/magdalenaschwarzl/Desktop/graph.png";
    //pix.save(path);
    painter->restore();
}
