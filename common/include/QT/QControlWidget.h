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

#ifndef QT_QCONTROLWIDGET_H
#define QT_QCONTROLWIDGET_H

#include <QWidget>
#include <QGridLayout>
#include <QRadioButton>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QGroupBox>
#include <QMenuBar>
#include <QSlider>

#include <QT/QDoubleSlider.h>
#include <QT/QExtendedVerticalSlider.h>

namespace QT {
    class QGroupControlWidget;

    class QControlWidget : public QWidget {
        Q_OBJECT

    protected:
        QGridLayout * layout;
        int row;
        QMenuBar * mb;

    public:
        QControlWidget( QWidget * parent = 0 );

        QRadioButton        * addRadioButton            ( QString title );
        QSpinBox            * addIntSpinBox             ( QString label );
        QDoubleSpinBox      * addDblSpinBox             ( QString label );
        QCheckBox           * addCheckbox               ( QString label );
        QPushButton         * addButton                 ( QString label );
        QGroupControlWidget * addGroupControlWidget     ( QString title );
        QMenuBar            * menuBar();
        QSlider             * addHorizontalSlider       ( QString label );
        //QSlider             * addVerticalSlider         ( QString label );
        QExtendedVerticalSlider *addVerticalSlider         ( QString label );
        QDoubleSlider       * addHorizontalDoubleSlider ( QString label );
        QDoubleSlider       * addVerticalDoubleSlider   ( QString label );

    };

    class QGroupControlWidget : public QGroupBox {
        Q_OBJECT

    protected:
        QGridLayout * layout;
        int row;

    public:
        QGroupControlWidget( QString title, QWidget * parent = 0 );

        QRadioButton   * addRadioButton( QString title );
        QDoubleSpinBox * addSpinBox    ( QString label );
        QCheckBox      * addCheckbox   ( QString label );
        QPushButton    * addButton     ( QString label );

    };

}

#endif // QT_QCONTROLWIDGET_H
