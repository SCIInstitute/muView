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

#ifndef QT_QEXTENDEDGLWIDGET_H
#define QT_QEXTENDEDGLWIDGET_H

#include <QGLWidget>

namespace QT {
    class QExtendedGLWidget : public QGLWidget
    {
        Q_OBJECT

    public:

        QExtendedGLWidget( QWidget *parent );
        ~QExtendedGLWidget();

        virtual QSize minimumSizeHint() const { return QSize(   50,  50); }
        virtual QSize sizeHint()        const { return QSize( 1280, 720); }


        virtual void MouseDoubleClick( int mouse_button, int mouse_x, int mouse_y );
        virtual void MouseMove( int mouse_button, int mouse_x, int mouse_dx, int mouse_y, int mouse_dy );
        virtual void MousePress( int mouse_button, int mouse_x, int mouse_y );
        virtual void MouseRelease( int mouse_x, int mouse_y );

    public:
        virtual void mouseDoubleClickEvent ( QMouseEvent * event );
        virtual void mouseMoveEvent ( QMouseEvent * event );
        virtual void mousePressEvent ( QMouseEvent * event );
        virtual void mouseReleaseEvent ( QMouseEvent * event );
        //virtual void keyPressEvent ( QKeyEvent * event );
        //virtual void keyReleaseEvent ( QKeyEvent * event );

    private:
        int mouse_button, mouse_x, mouse_y;
        bool mouse_active;
    };

}


#endif // QT_QEXTENDEDGLWIDGET_H
