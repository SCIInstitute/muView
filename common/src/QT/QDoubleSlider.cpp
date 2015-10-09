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

#include <QT/QDoubleSlider.h>

using namespace QT;

QDoubleSlider::QDoubleSlider( QWidget * parent ) : QSlider(parent) {
    connect( this, SIGNAL(valueChanged(int)), this, SLOT(updateValue(int)));
    QSlider::setRange( 0, 1000 );
}

QDoubleSlider::QDoubleSlider( Qt::Orientation orientation, QWidget * parent ) : QSlider(orientation,parent){
    connect( this, SIGNAL(valueChanged(int)), this, SLOT(updateValue(int)));
    QSlider::setRange( 0, 1000 );
}

void QDoubleSlider::updateValue(int value){
    double t = (double)value/1000.0;
    emit valueChanged( range_min + t*(range_max-range_min) );
}

void QDoubleSlider::setRange( double v_min, double v_max ){
    range_min = v_min;
    range_max = v_max;
    emit rangeChanged( range_min, range_max );
}

void QDoubleSlider::setValue( double val ){
    QSlider::setValue( (int)( 1000.0 * (val-range_min) / (range_max-range_min) ) );
}

double QDoubleSlider::getValue(){
    double t = (double)QSlider::value()/1000.0;
    return range_min + t*(range_max-range_min);
}

