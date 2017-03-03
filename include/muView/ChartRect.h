
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
    ChartRect(const QPointF &position, const SCI::Vex3 location, float* data, int numData);

    void drawChartRect(QPainter *painter, QPixmap pix, int w, int h);

    float width();
    float height();
    void resize(int w, int h);

    QPixmap grabChartView();


private:
    QPointF position;   // position on the screen -> 2D
    SCI::Vex3 location; // which vertex data we are showing -> 3D

    QChartView* chartView;
    std::vector<float> data; // from pdata at position
};

#endif
