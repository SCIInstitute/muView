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

#ifndef QPLUGINDOCK_H
#define QPLUGINDOCK_H

#include <QDockWidget>
#include <QT/QSplitterPlugin.h>

class QPluginDock : public QDockWidget {

    Q_OBJECT

public:
    QPluginDock( QString title, QSplitterPlugin * elem, QWidget * parent );

protected:
    virtual void showEvent ( QShowEvent * event );
    virtual void hideEvent ( QHideEvent * event );

    virtual void focusInEvent ( QFocusEvent * event );
    virtual void focusOutEvent ( QFocusEvent * event );

    virtual void closeEvent ( QCloseEvent * event );

};

#endif // QPLUGINDOCK_H
