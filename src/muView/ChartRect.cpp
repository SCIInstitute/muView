#include <QtGui>

#include "muView/ChartRect.h"

ChartRect::ChartRect(QPointF &position, float *data, int numData, int w, int h)
    : position(position){

    QLineSeries *series = new QLineSeries();

    for (int i=0; i<numData;i++){
        this->data.push_back(data[i]);
        *series << QPointF(i, data[i]);
    }


    QChart *chart = new QChart();

    chart->legend()->hide();
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->setMargins(QMargins(1,1,1,1));//(int left, int top, int right, int bottom)




    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    resize(w,h);
}

QPixmap ChartRect::grabChartView(){
    return this->chartView->grab();
}


void ChartRect::setData(float *data, int numData){

    chartView->chart()->removeAllSeries();

    QLineSeries *series = new QLineSeries();
    for (int i=0; i<numData;i++){
        this->data.push_back(data[i]);
        *series << QPointF(i, data[i]);
    }

    chartView->chart()->addSeries(series);
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
    QString path = "/Users/magdalenaschwarzl/Desktop/graph.png";
    pix.save(path);
    painter->restore();
}
