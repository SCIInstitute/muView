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

#include <QT/QSplitterPlugin.h>

#include <stdio.h>

QSplitterPlugin::QSplitterPlugin(QMenuBar * _menu_bar, QWidget * parent) : QSplitter(Qt::Horizontal, parent) {
    menu_bar   = _menu_bar;

    //setFocusPolicy( Qt::ClickFocus );
}

QSplitterPlugin::~QSplitterPlugin(){
    while(object_list.size()>0){
        //delete object_list.back();
        object_list.pop_back();
    }
}


void QSplitterPlugin::showEvent ( QShowEvent * ){
    AttachMenus();
}

void QSplitterPlugin::hideEvent ( QHideEvent * ){
    DetachMenus();
}

void QSplitterPlugin::keyPressEvent ( QKeyEvent * ){

}

void QSplitterPlugin::keyReleaseEvent ( QKeyEvent * ){

}


void QSplitterPlugin::focusInEvent ( QFocusEvent * ){
    printf("focus in\n");fflush(stdout);
}

void QSplitterPlugin::focusOutEvent ( QFocusEvent * ){
    printf("focus out\n");fflush(stdout);
}

void QSplitterPlugin::AttachMenus ( ){
    std::map<QMenu*,QAction*>::iterator iter;
    for(iter=menu_map.begin() ; iter != menu_map.end(); iter++){
        QMenu* menu = iter->first;
        if( menu_map[menu] == 0 ){
            menu_map[menu] = menu_bar->addMenu( menu );
        }
    }
}

void QSplitterPlugin::DetachMenus ( ){
    std::map<QMenu*,QAction*>::iterator iter;
    for(iter=menu_map.begin() ; iter != menu_map.end(); iter++){
        QMenu* menu = iter->first;
        if( menu_map[menu] != 0 ){
            menu_bar->removeAction( menu_map[menu] );
            menu_map[menu] = 0;
        }
    }
}

QMenu * QSplitterPlugin::AddMenu( QString title ){
    QMenu * menu = new QMenu( title );
    menu_map[menu] = 0;
    object_list.push_back(menu);
    return menu;
}

QAction * QSplitterPlugin::AddAction( QString title ){
    QAction * action = new QAction( title, this );
    object_list.push_back(action);
    return action;
}

QAction * QSplitterPlugin::AddAction( QString title, QVariant data ){
    QAction * action = new QAction( title, this );
    action->setData( data );
    object_list.push_back(action);
    return action;
}

QActionGroup * QSplitterPlugin::AddActionGroup( ){
    QActionGroup * ret = new QActionGroup(this);
    object_list.push_back(ret);
    return ret;
}


