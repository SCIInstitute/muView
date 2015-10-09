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

#include <QT/QExtendedGLWidget.h>

#include <QMouseEvent>

using namespace QT;

QExtendedGLWidget::QExtendedGLWidget( QWidget *parent) : QGLWidget( QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer | QGL::Rgba | QGL::AlphaChannel | QGL::DirectRendering | QGL::SampleBuffers), parent ) {

    mouse_active = false;
    mouse_button = Qt::NoButton;
    mouse_x = 0;
    mouse_y = 0;

    setMouseTracking(true);

}

QExtendedGLWidget::~QExtendedGLWidget(){ }


void QExtendedGLWidget::mouseDoubleClickEvent ( QMouseEvent * event ){

    mouse_button = event->button();
    mouse_x = event->x();
    mouse_y = size().height()-event->y()-1;

    MouseDoubleClick( mouse_button, mouse_x, mouse_y );

}

void QExtendedGLWidget::mouseMoveEvent ( QMouseEvent * event ){

    int dx = event->x()-mouse_x;
    int dy = (size().height()-event->y()-1)-mouse_y;

    mouse_x = event->x();
    mouse_y = size().height()-event->y()-1;

    MouseMove( mouse_button, mouse_x, dx, mouse_y, dy );

}

void QExtendedGLWidget::mousePressEvent ( QMouseEvent * event ){

    mouse_button = event->button();
    mouse_x = event->x();
    mouse_y = size().height()-event->y()-1;

    MousePress( mouse_button, mouse_x, mouse_y );

}

void QExtendedGLWidget::mouseReleaseEvent ( QMouseEvent * event ){

    mouse_button = Qt::NoButton;
    mouse_x = event->x();
    mouse_y = size().height()-event->y()-1;

    MouseRelease( mouse_x, mouse_y );

}


void QExtendedGLWidget::MouseDoubleClick( int , int , int ){ }
void QExtendedGLWidget::MouseMove( int , int , int , int , int ){ }
void QExtendedGLWidget::MousePress( int , int , int  ){ }
void QExtendedGLWidget::MouseRelease( int , int  ){ }

//void MainWidget::keyPressEvent   ( QKeyEvent * event ){ }
//void MainWidget::keyReleaseEvent ( QKeyEvent * event ){ }



