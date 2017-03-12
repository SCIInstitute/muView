#include <QtGui>

#include "muView/ChartRect.h"

ChartRect::ChartRect(const QPointF &position, const SCI::Vex3 location, float *data, int numData)
    : position(position), location(location){

    QLineSeries *series = new QLineSeries();

    for (int i=0; i<numData;i++){
        this->data.push_back(data[i]);
        *series << QPointF(i, data[i]);
    }


    QChart *chart = new QChart();

    chart->legend()->hide();
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->setMargins(QMargins(2,2,2,2));//(int left, int top, int right, int bottom)



    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
}

QPixmap ChartRect::grabChartView(){
    return this->chartView->grab();
}


void ChartRect::resize(int w, int h){
    this->chartView->resize(w, h);
}

float ChartRect::width(){
    return this->chartView->size().width();
}

float ChartRect::height(){
    return this->chartView->size().height();
}


void ChartRect::drawChartRect(QPainter *painter, QPixmap pix, int w, int h)
{

    std::cout << " x:" << position.x() << " y:" << position.y() << std::endl;
    painter->save();
    painter->drawPixmap(position.x(), position.y(), w, h, pix);
    QImage * img = new QImage();
    *img = QImage(w*50,h*50, QImage::Format_ARGB32);
    QString path = "/Users/magdalenaschwarzl/Desktop/graph.png";
    pix.save(path);
    painter->restore();
}
