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

#include <Dialog/Connection.h>
#include <iostream>
#include <sstream>

using namespace Dialog;

Connection::Connection(QWidget *parent) : QDialog(parent) {

    success = false;

    hostLabel = new QLabel(tr("&Server name:"));
    portLabel = new QLabel(tr("S&erver port:"));

    hostLineEdit = new QLineEdit( tr( tcpSocket.GetIPAddress().c_str() ) );
    portLineEdit = new QLineEdit( tr("3000") );
    portLineEdit->setValidator(new QIntValidator(1, 65535, this));

    hostLabel->setBuddy(hostLineEdit);
    portLabel->setBuddy(portLineEdit);

    statusLabel = new QLabel(tr("Not connected."));

    connectButton = new QPushButton(tr("Connect"));
    connectButton->setDefault(true);
    connectButton->setEnabled(true);

    quitButton = new QPushButton(tr("Quit"));

    buttonBox = new QDialogButtonBox;
    buttonBox->addButton(connectButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(quitButton,    QDialogButtonBox::RejectRole);

    QGridLayout *mainLayout = new QGridLayout;
    mainLayout->addWidget(hostLabel,    0, 0);
    mainLayout->addWidget(hostLineEdit, 0, 1);
    mainLayout->addWidget(portLabel,    1, 0);
    mainLayout->addWidget(portLineEdit, 1, 1);
    mainLayout->addWidget(statusLabel,  2, 0, 1, 2);
    mainLayout->addWidget(buttonBox,    3, 0, 1, 2);
    setLayout(mainLayout);

    setWindowTitle(tr("Server Connection"));

    connect(hostLineEdit,  SIGNAL(textChanged(const QString &)), this, SLOT(enableConnectButton()) );
    connect(portLineEdit,  SIGNAL(textChanged(const QString &)), this, SLOT(enableConnectButton()) );
    connect(connectButton, SIGNAL(clicked()),                    this, SLOT(connectToServer())     );
    connect(quitButton,    SIGNAL(clicked()),                    this, SLOT(close())               );

    connect(&tcpSocket, SIGNAL(connected()), this, SLOT(connectedToServer()));
    connect(&tcpSocket, SIGNAL(readyRead()), this, SLOT(readDatasetList()));
    connect(&tcpSocket, SIGNAL(SocketError(QTcpSocket::SocketError)), this, SLOT(displayError(QTcpSocket::SocketError)));

}

void Connection::connectToServer() {

    connectButton->setEnabled(false);
    tcpSocket.abort();
    tcpSocket.connectToHost(hostLineEdit->text(), portLineEdit->text().toInt());
    QTimer::singleShot(10000, this, SLOT(checkForError()));

}

void Connection::checkForError(){
    if(!success){
        tcpSocket.abort();
        displayError( QTcpSocket::HostNotFoundError );
    }
}

void Connection::connectedToServer(){
    tcpSocket.Send( "servers" );
    tcpSocket.Send( "disconnect" );
    tcpSocket.flush();
}

void Connection::readDatasetList() {

    statusLabel->setText("Connected");
    connectButton->setEnabled(false);

    int dataCount = tcpSocket.RecvInt( );
    for(int i = 0; i < dataCount; i++){
        std::string name = tcpSocket.RecvString( );
        std::string type = tcpSocket.RecvString( );
        std::string ip   = tcpSocket.RecvString( );
        short       port = tcpSocket.RecvShort( );

        std::stringstream ss;
        ss << name << " (" << type << ") @ " << ip << ":" << port;
        dataset_list.push_back( ss.str() );
        dataset_ip.push_back( ip );
        dataset_port.push_back( port );
    }
    tcpSocket.disconnectFromHost();

    QTimer::singleShot(500, this, SLOT(close()));

    success = true;

}

bool Connection::isSuccessful(){
    return success;
}

void Connection::enableConnectButton() {
    connectButton->setEnabled( !hostLineEdit->text().isEmpty() && !portLineEdit->text().isEmpty());
}

 void Connection::displayError(QTcpSocket::SocketError socketError) {

     switch (socketError) {
     case QTcpSocket::RemoteHostClosedError:
         break;
     case QTcpSocket::HostNotFoundError:
         QMessageBox::information(this, tr("Server Connection"),
                                  tr("The host was not found. Please check the "
                                     "host name and port settings."));
         break;
     case QTcpSocket::ConnectionRefusedError:
         QMessageBox::information(this, tr("Server Connection"),
                                  tr("The connection was refused by the peer. "
                                     "Make sure the server is running, "
                                     "and check that the host name and port "
                                     "settings are correct."));
         break;
     default:
         QMessageBox::information(this, tr("Server Connection"),
                                  tr("The following error occurred: %1.")
                                  .arg(tcpSocket.errorString()));
     }

     connectButton->setEnabled(true);
 }

