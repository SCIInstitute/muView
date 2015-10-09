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

#ifndef QSPLITTERPLUGIN_H
#define QSPLITTERPLUGIN_H

#include <QSplitter>
#include <QtOpenGL/QGLWidget>
#include <QT/QMultipleSourceCodeWidget.h>
#include <QMenuBar>

class QSplitterPlugin  : public QSplitter {

    Q_OBJECT

public:
    QSplitterPlugin(QMenuBar * _menu_bar, QWidget * parent);
    ~QSplitterPlugin();

    virtual void AttachMenus ( );
    virtual void DetachMenus ( );

    virtual void showEvent ( QShowEvent * event );
    virtual void hideEvent ( QHideEvent * event );

    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void keyReleaseEvent ( QKeyEvent * event );

protected:

    virtual void	focusInEvent ( QFocusEvent * event );
    //bool	focusNextChild ()
    //virtual bool	focusNextPrevChild ( bool next )
    virtual void	focusOutEvent ( QFocusEvent * event );

protected:

    QMenu        * AddMenu( QString title );
    QAction      * AddAction( QString title );
    QAction      * AddAction( QString title, QVariant data );
    QActionGroup * AddActionGroup( );

    std::map<QMenu*,QAction*>   menu_map;

    QMenuBar                  * menu_bar;

    std::vector<void *> object_list;


};

#endif // QPLUGIN_H
