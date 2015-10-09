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

#include <QT/QPluginDock.h>

#include <stdio.h>

QPluginDock::QPluginDock( QString title, QSplitterPlugin * elem, QWidget * parent ) : QDockWidget( title, parent ) {

    setWidget( elem );
    setFocusPolicy( Qt::ClickFocus );

}


void QPluginDock::showEvent ( QShowEvent * ){
    //printf("dock - show\n"); fflush(stdout);
}

void QPluginDock::hideEvent ( QHideEvent * ){
    //printf("dock - hide\n"); fflush(stdout);
}

void QPluginDock::focusInEvent ( QFocusEvent * ){
    //printf("dock - focus in\n"); fflush(stdout);
}

void QPluginDock::focusOutEvent ( QFocusEvent * ){
    //printf("dock - focus out\n"); fflush(stdout);
}

void QPluginDock::closeEvent ( QCloseEvent * ){
    //printf("dock - close\n"); fflush(stdout);
}
