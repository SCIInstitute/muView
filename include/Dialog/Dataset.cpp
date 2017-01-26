/****************************************************************************
 **
 ** Copyright (C) 2009 Nokia Corporation and/or its subsidiary(-ies).
 ** All rights reserved.
 ** Contact: Nokia Corporation (qt-info@nokia.com)
 **
 ** This file is part of the examples of the Qt Toolkit.
 **
 ** $QT_BEGIN_LICENSE:LGPL$
 ** Commercial Usage
 ** Licensees holding valid Qt Commercial licenses may use this file in
 ** accordance with the Qt Commercial License Agreement provided with the
 ** Software or, alternatively, in accordance with the terms contained in
 ** a written agreement between you and Nokia.
 **
 ** GNU Lesser General Public License Usage
 ** Alternatively, this file may be used under the terms of the GNU Lesser
 ** General Public License version 2.1 as published by the Free Software
 ** Foundation and appearing in the file LICENSE.LGPL included in the
 ** packaging of this file.  Please review the following information to
 ** ensure the GNU Lesser General Public License version 2.1 requirements
 ** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 **
 ** In addition, as a special exception, Nokia gives you certain additional
 ** rights.  These rights are described in the Nokia Qt LGPL Exception
 ** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
 **
 ** GNU General Public License Usage
 ** Alternatively, this file may be used under the terms of the GNU
 ** General Public License version 3.0 as published by the Free Software
 ** Foundation and appearing in the file LICENSE.GPL included in the
 ** packaging of this file.  Please review the following information to
 ** ensure the GNU General Public License version 3.0 requirements will be
 ** met: http://www.gnu.org/copyleft/gpl.html.
 **
 ** If you have questions regarding the use of this file, please contact
 ** Nokia at qt-info@nokia.com.
 ** $QT_END_LICENSE$
 **
 ****************************************************************************/

 #include <QtGui>
 #include <QtNetwork>

#include <SCI/Vex2.h>

 #include <Dialog/Dataset.h>
#include <iostream>


using namespace Dialog;

Dataset::Dataset(std::vector<std::string> & dataset_list, QWidget *parent) : QDialog(parent) {

     vbox = new QVBoxLayout;
     for(int i = 0; i < (int)dataset_list.size(); i++){
         datasetButtons.push_back( new QRadioButton(tr(dataset_list[i].c_str())) );
         if(i==0) datasetButtons.back()->setChecked(true);
         vbox->addWidget(datasetButtons.back());
     }
     vbox->addStretch(1);

     groupBox = new QGroupBox(tr("Available Datasets"));
     groupBox->setLayout(vbox);

     selectButton = new QPushButton(tr("Select"));
     selectButton->setDefault(true);
     selectButton->setEnabled(true);

     quitButton = new QPushButton(tr("Quit"));

     buttonBox = new QDialogButtonBox;
     buttonBox->addButton(selectButton, QDialogButtonBox::ActionRole);
     buttonBox->addButton(quitButton,   QDialogButtonBox::RejectRole);

     QGridLayout *mainLayout = new QGridLayout;
     mainLayout->addWidget(buttonBox, 1, 0, 1, 2);
     mainLayout->addWidget(groupBox, 0, 0 );
     setLayout(mainLayout);

     connect(selectButton, SIGNAL(clicked()), this, SLOT(selectDataset()));
     connect(selectButton, SIGNAL(clicked()), this, SLOT(close()));
     connect(quitButton,   SIGNAL(clicked()), this, SLOT(close()));

     setWindowTitle(tr("Dataset Selection"));

     selected = -1;
}

void Dataset::selectDataset( ){
    selected = -1;
    for(int i = 0; i < (int)datasetButtons.size(); i++){
        if( datasetButtons[i]->isChecked() ) selected = i;
    }
}

int Dataset::GetSelected( ){ return selected; }


