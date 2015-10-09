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

#ifndef QT_QDOUBLESLIDER_H
#define QT_QDOUBLESLIDER_H

#include <QWidget>
#include <QGridLayout>
#include <QRadioButton>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QGroupBox>
#include <QMenuBar>
#include <QSlider>

namespace QT {
    class QDoubleSlider : public QSlider {
        Q_OBJECT


    signals:
        void valueChanged(double);
        void rangeChanged(double,double);

    private slots:
        void updateValue(int);

    public:
        QDoubleSlider( QWidget * parent = 0 );
        QDoubleSlider( Qt::Orientation orientation, QWidget * parent = 0 );

        void setRange( double v_min, double v_max );
        void setValue( double val );
        double getValue();

    protected:
        double range_min, range_max;
    };

}

#endif // QT_QDOUBLESLIDER_H
