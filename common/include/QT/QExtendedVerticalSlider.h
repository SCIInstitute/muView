/*
**  Common QT Extensions Library
**  Copyright (C) 2013  Paul Rosen
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef QT_QEXTENDEDVERTICALSLIDER_H
#define QT_QEXTENDEDVERTICALSLIDER_H

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QGroupBox>

namespace QT {

    class QExtendedVerticalSlider : public QGroupBox {

        Q_OBJECT

    public:

        QExtendedVerticalSlider(QString label, QWidget * parent = 0 );
        ~QExtendedVerticalSlider();

        void setRange( int range_min, int range_max );
        void setRange( double range_min, double range_max );

        void setValue( int value );
        void setValue( double value );

    signals:
        void valueChanged(int value);
        void rangeChanged(int range_min, int range_max);

        void valueChanged(double value);
        void rangeChanged(double range_min, double range_max);

    private slots:
        void updateValue(int);

    protected:
        QLabel  * lbl_min;
        QLabel  * lbl_max;
        QSlider * sld;
        QLabel  * lbl_val;
        QLabel  * lbl_curr;

        bool double_toggle;

        double rng_min, rng_max;

    };
}


#endif // QT_QEXTENDEDVERTICALSLIDER_H
