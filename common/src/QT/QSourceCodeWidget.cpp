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

#include <QT/QSourceCodeWidget.h>
#include <QFileInfo>


#if __APPLE__

    QSourceCodeWidget::QSourceCodeWidget( int _index, const char * fname, const char * srcTxt ) : QTextEdit( 0 ) {

        index      = _index;
        long_name  = QString(fname);
        short_name = QFileInfo(long_name).fileName();

        QFont font(tr("Courier New"));
        font.setPointSize(8);

        setFont( font );

        setLineWrapMode( NoWrap );
        setText( QString(srcTxt) );
        setReadOnly(true);

        current_marker=0;

    }

    QSourceCodeWidget::~QSourceCodeWidget( ){ }

    int QSourceCodeWidget::CreateMarkerCircle( QColor color, int markerID ){    return 0;   }

    int QSourceCodeWidget::CreateMarkerBackground( QColor color, int markerID ){    return 0;   }

    void QSourceCodeWidget::MarkLine( int markerID, int lineno ){   }

    void QSourceCodeWidget::UnmarkLine( int markerID, int lineno ){ }

    void QSourceCodeWidget::UnmarkAll( int markerID ){  }

    void QSourceCodeWidget::SetCursorLine( int lineno ){    }

    int QSourceCodeWidget::GetIndex( ){     return index;   }

#else

    QSourceCodeWidget::QSourceCodeWidget( int _index, const char * fname, const char * srcTxt ) : QsciScintilla ( 0 ) {

        index      = _index;
        long_name  = QString(fname);
        short_name = QFileInfo(long_name).fileName();

        QFont font(tr("Courier New"));
        font.setPointSize(8);

        setLexer(new QsciLexerCPP);
        lexer()->setFont(font);

        setText( QString(srcTxt) );
        setReadOnly(true);

        setFolding(QsciScintilla::BoxedTreeFoldStyle);

        setMarginType(1,QsciScintilla::NumberMargin);
        setMarginWidth(1,30);
        setMarginSensitivity(1,true);

        current_marker=0;

    }

    QSourceCodeWidget::~QSourceCodeWidget( ){
        if(lexer())
            delete lexer();
    }

    int QSourceCodeWidget::CreateMarkerCircle( QColor color, int markerID ){
        markerID = markerDefine( QsciScintilla::Circle, markerID );
        if(markerID==-1){
            printf("Marker Creation Failed\n"); fflush(stdout);
        }
        else{
            setMarkerBackgroundColor( color, markerID );
            setMarkerForegroundColor( color, markerID );
        }
        return markerID;
    }

    int QSourceCodeWidget::CreateMarkerBackground( QColor color, int markerID ){
        markerID = markerDefine( QsciScintilla::Background, markerID );
        if(markerID==-1){
            printf("Marker Creation Failed\n"); fflush(stdout);
        }
        else{
            setMarkerBackgroundColor( color, markerID );
            setMarkerForegroundColor( QColor(  0,  0,  0), markerID );
        }
        return markerID;
    }

    void QSourceCodeWidget::MarkLine( int markerID, int lineno ){
        markerAdd( lineno-1, markerID );
    }

    void QSourceCodeWidget::UnmarkLine( int markerID, int lineno ){
        markerDelete( lineno-1, markerID );
    }

    void QSourceCodeWidget::UnmarkAll( int markerID ){
        markerDeleteAll( markerID );
    }

    void QSourceCodeWidget::SetCursorLine( int lineno ){
        setCursorPosition(lineno-1,0);
    }

    int QSourceCodeWidget::GetIndex( ){
        return index;
    }

#endif
