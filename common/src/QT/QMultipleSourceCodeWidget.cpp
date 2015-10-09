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

#include <QT/QMultipleSourceCodeWidget.h>
#include <QFileInfo>


QMultipleSourceCodeWidget::QMultipleSourceCodeWidget( QWidget * parent) : QWidget(parent){

    connect( &cbox, SIGNAL(currentIndexChanged(int)), this, SLOT(change_file(int)) );

    lyout.setSpacing(0);
    lyout.setContentsMargins(0,0,0,0);
    lyout.addWidget( &cbox );
    setLayout( &lyout );

}

QMultipleSourceCodeWidget::~QMultipleSourceCodeWidget(){

    for(int i = 0; i < (int)src.size(); i++){
        delete src[i];
    }

}


QSize QMultipleSourceCodeWidget::minimumSizeHint() const { return QSize(50, 50); }

QSize QMultipleSourceCodeWidget::sizeHint() const { return QSize(400, 1200); }


void QMultipleSourceCodeWidget::AddFile( int index, const char * _fname, const char * srcTxt ){

    QString srcText;

    if(srcTxt){
        srcText = QString(srcTxt);
    }
    else{
        QFile file( tr(_fname) );
        if (!file.open(QFile::ReadOnly)) {
            return;
        }
        srcText = QString(srcTxt);
        file.close();
    }

    src.push_back( new QSourceCodeWidget( index, _fname, srcTxt ) );
    cbox.insertItem( cbox.count(), src.back()->GetShortName() );
    lyout.addWidget( src.back() );

    if(src.size()!=1){
        src.back()->hide();
    }

}

QSourceCodeWidget * QMultipleSourceCodeWidget::Find( int idx ){

    for(int i = 0; i < (int)src.size(); i++){
        if( src[i]->GetIndex() == idx ) return src[i];
    }
    return 0;

}

void QMultipleSourceCodeWidget::CreateMarkerCircle( QColor color, int markerID ){
    for(int i = 0; i < (int)src.size(); i++){
        src[i]->CreateMarkerCircle( color, markerID );
    }
}

void QMultipleSourceCodeWidget::CreateMarkerBackground( QColor color, int markerID ){
    for(int i = 0; i < (int)src.size(); i++){
        src[i]->CreateMarkerBackground( color, markerID );
    }
}


void QMultipleSourceCodeWidget::change_file(int idx){

    for(int i = 0; i < (int)src.size(); i++){
        if( i == idx ){ src[i]->show(); }
        else{           src[i]->hide(); }
    }

}

