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

#include <QT/QExtendedVerticalSlider.h>

#include <QGridLayout>

QT::QExtendedVerticalSlider::QExtendedVerticalSlider(QString label, QWidget *parent) : QGroupBox( label, parent) {

    QGridLayout * layout = new QGridLayout( );

    lbl_min  = new QLabel( );
    lbl_max  = new QLabel( );
    sld      = new QSlider( Qt::Vertical );
    lbl_val  = new QLabel( tr("<center>Current<br>Value</center>") );
    lbl_curr = new QLabel( );

    double_toggle = false;

    rng_min = 0;
    rng_max = 1;

    updateValue( sld->value() );
    setRange( sld->minimum(), sld->maximum() );

    connect(sld,SIGNAL(valueChanged(int)),this,SLOT(updateValue(int)));

    layout->addWidget(sld,      0, 0, 5, 1, Qt::AlignRight );
    layout->addWidget(lbl_max,  0, 1, 1, 1, Qt::AlignLeft | Qt::AlignTop );
    layout->addWidget(lbl_min,  4, 1, 1, 1, Qt::AlignLeft | Qt::AlignBottom );

    layout->addWidget(lbl_val,  1, 2, 2, 1, Qt::AlignCenter );
    layout->addWidget(lbl_curr, 3, 2, 1, 1, Qt::AlignCenter );

    setLayout(layout);
}

QT::QExtendedVerticalSlider::~QExtendedVerticalSlider(){

}

void QT::QExtendedVerticalSlider::setValue( int value ){
    double_toggle = false;
    sld->setValue(value);
}

void QT::QExtendedVerticalSlider::setValue( double value ){
    double_toggle = true;
    int v = (int)((value-rng_min)/(rng_max-rng_min)*1000.0);
    sld->setValue(v);
}


void QT::QExtendedVerticalSlider::updateValue(int val){
    if( double_toggle ){
        double dval = rng_min + (rng_max-rng_min)*(double)val/1000.0;
        lbl_curr->setText( QString("%1").arg(dval,0,'g',4) );
        emit valueChanged(dval);
    }
    else{
        lbl_curr->setText( QString("%1").arg(val) );
        emit valueChanged(val);
    }
}

void QT::QExtendedVerticalSlider::setRange( double range_min, double range_max ){
    double_toggle = true;
    sld->setRange( 0, 1000 );
    rng_min = range_min;
    rng_max = range_max;
    lbl_min->setText( QString("%1").arg(range_min) );
    lbl_max->setText( QString("%1").arg(range_max) );
    emit rangeChanged(range_min,range_max);
}

void QT::QExtendedVerticalSlider::setRange( int range_min, int range_max ){
    double_toggle = false;
    sld->setRange( range_min, range_max );
    lbl_min->setText( QString("%1").arg(range_min) );
    lbl_max->setText( QString("%1").arg(range_max) );
    emit rangeChanged(range_min,range_max);
}



