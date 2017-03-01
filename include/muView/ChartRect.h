
#ifndef ChartRect_H
#define ChartRect_H

#include <QBrush>
#include <QColor>
#include <QPointF>
#include <QRect>
#include <QRectF>

class QPainter;

class ChartRect
{
public:
    ChartRect(const QPointF &position);

    void drawChartRect(QPainter *painter, QPixmap pix, int w, int h);

private:
    QPointF position;
};

#endif
