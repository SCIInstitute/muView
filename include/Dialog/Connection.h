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

 #ifndef DIALOG_CONNECTION_H
 #define DIALOG_CONNECTION_H

#include <QDialog>
#include <QTcpSocket>
#include <QGridLayout>

#include <SCI/Network/QExtendedTcpSocket.h>

 class QDialogButtonBox;
 class QLabel;
 class QLineEdit;
 class QPushButton;
 class QTcpSocket;

 namespace Dialog {
     class Connection : public QDialog
     {
         Q_OBJECT

     public:
         Connection(QWidget *parent = 0);

         bool isSuccessful();

         std::vector<std::string> dataset_ip;
         std::vector< short >     dataset_port;
         std::vector<std::string> dataset_list;

     protected:
         bool success;

     private slots:
         void connectToServer();
         void readDatasetList();
         void displayError(QTcpSocket::SocketError socketError);
         void enableConnectButton();
         void checkForError();
         void connectedToServer();

     private:
         QLabel *hostLabel;
         QLabel *portLabel;

         QLineEdit *hostLineEdit;
         QLineEdit *portLineEdit;

         QLabel *statusLabel;

         QPushButton *connectButton;
         QPushButton *quitButton;

         QDialogButtonBox *buttonBox;
         QGridLayout *mainLayout;

         SCI::Network::QExtendedTcpSocket tcpSocket;
         QString currentFortune;
         quint16 blockSize;

     };

 }

 #endif
