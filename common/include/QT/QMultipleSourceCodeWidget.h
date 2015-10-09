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

#ifndef QMULTIPLESOURCECODEWIDGET_H
#define QMULTIPLESOURCECODEWIDGET_H

#include <QDockWidget>
#include <QVBoxLayout>
#include <QComboBox>
#include <QT/QSourceCodeWidget.h>

#include <vector>

class QMultipleSourceCodeWidget : public QWidget
{
    Q_OBJECT

public:
    QMultipleSourceCodeWidget(QWidget * parent = 0);
    ~QMultipleSourceCodeWidget();

    void AddFile( int index, const char * fname, const char * srcTxt = 0 );
    QSourceCodeWidget * Find( int idx );

    void CreateMarkerCircle( QColor color, int markerID = -1 );
    void CreateMarkerBackground( QColor color, int markerID = -1 );

    QSize minimumSizeHint() const;
    QSize sizeHint() const;


public slots:
    void change_file(int);

protected:
    QComboBox   cbox ;
    QVBoxLayout lyout;

    std::vector<QSourceCodeWidget *> src;

};



#endif // QMULTIPLESOURCECODEWIDGET_H
