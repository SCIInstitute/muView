#include <QtGui>

#include "muView/ChartRect.h"

ChartRect::ChartRect(const QPointF &position)
    : position(position){}



void ChartRect::drawChartRect(QPainter *painter, QPixmap pix, int w, int h)
{
    painter->save();
    painter->drawPixmap(position.x(), position.y(), w, h, pix);
    painter->restore();
}
