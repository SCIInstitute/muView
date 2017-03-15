
#ifndef ChartRect_H
#define ChartRect_H

#include <QBrush>
#include <QColor>
#include <QPointF>
#include <QRect>
#include <QRectF>
#include <QPixmap>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChart>

using namespace QtCharts;

#include <SCI/Camera/FrustumProjection.h>
#include <SCI/Camera/OrthoProjection.h>
#include <SCI/Camera/ThirdPersonCameraControls.h>

class QPainter;

class ChartRect
{
public:
    ChartRect(QPointF &position, float* data, int numData, int w, int h, float min, float max, float steps);

    void drawChartRect(QPainter *painter, QPixmap pix);

    float width();
    float height();
    void resize(int w, int h);

    QPixmap grabChartView();

    void setLocation(SCI::Vex3 location);
    void setPosition(QPointF pos);
    void setData(float* data, int numData);

    void computeBinning(float min, float max, int steps);


private:
    QPointF position;   // position on the screen -> 2D
    SCI::Vex3 location; // which vertex data we are showing -> 3D

    int w,h;

    std::vector<float> axisTicks;

    QChartView* chartView;
    std::vector<float> data; // from pdata at position
};

#endif
