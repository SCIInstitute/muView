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

#ifndef QSOURCECODEWIDGET_H
#define QSOURCECODEWIDGET_H

#ifdef __APPLE__

    #include <QTextEdit>

    class QSourceCodeWidget : public QTextEdit

#else

    #include <Qsci/qsciscintilla.h>
    #include <Qsci/qscilexercpp.h>

    class QSourceCodeWidget : public QsciScintilla

#endif

    {
       Q_OBJECT

    public:
       QSourceCodeWidget( int index, const char * fname, const char * srcTxt );
       ~QSourceCodeWidget();

       virtual QSize minimumSizeHint() const { return QSize(200, 50); }
       virtual QSize sizeHint()        const { return QSize(550,550); }

       QString GetLongName(){ return long_name; }
       QString GetShortName(){ return short_name; }

       //void AddMarker( int lineno, QColor color );
       int CreateMarkerCircle( QColor color, int markerID = -1 );
       int CreateMarkerBackground( QColor color, int markerID = -1 );

       void MarkLine( int markerID, int lineno );
       void UnmarkLine( int markerID, int lineno );
       void UnmarkAll( int markerID );

       void SetCursorLine( int lineno );

       int GetIndex( );

    protected:

       QString long_name;
       QString short_name;

       int current_marker;
       int index;

    };


#endif // QSOURCECODEWIDGET_H

