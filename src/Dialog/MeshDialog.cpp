/*

#include <QtGui>
#include <QtNetwork>

#include <SCI/Vex2.h>

#include <Dialog/MeshDialog.h>
#include <iostream>
#include <sstream>

using namespace Dialog;

MeshDialog::MeshDialog(Data::ProxyData & _data, QWidget *parent) : QDialog(parent) {

   data = &_data;
   connN = -1;
   vertN = -1;


   std::cout << "Requesting MeshDialog from " << data->GetIP() << ":" << data->GetPort() << std::endl;
   fflush(stdout);

   success = false;

   statusLabel = new QLabel(tr("Connecting to Server"));

   quitButton = new QPushButton(tr("Cancel Connection"));
   buttonBox = new QDialogButtonBox;
   buttonBox->addButton(quitButton,    QDialogButtonBox::RejectRole);

   QGridLayout *mainLayout = new QGridLayout;
   mainLayout->addWidget(statusLabel,  0, 0, 1, 1);
   mainLayout->addWidget(buttonBox,    1, 0, 1, 1);
   setLayout(mainLayout);

   setWindowTitle(tr("Server Query (Geometry)"));

   connect(  quitButton, SIGNAL(clicked()),   this, SLOT(prematureClose()) );
   connect( &tcpSocket,  SIGNAL(connected()), this, SLOT(sendCommand()) );
   connect( &tcpSocket,  SIGNAL(readyRead()), this, SLOT(readData()) );
   connect( &tcpSocket,  SIGNAL(SocketError(QTcpSocket::SocketError)), this, SLOT(displayError(QTcpSocket::SocketError)));

   tcpSocket.abort();
   tcpSocket.connectToHost( tr(data->GetIP().c_str()), data->GetPort() );
   QTimer::singleShot(30000, this, SLOT(checkForError()));

}


void MeshDialog::checkForError(){
   if(!success){
       tcpSocket.abort();
       displayError( QTcpSocket::HostNotFoundError );
   }
}

void MeshDialog::sendCommand(){
   statusLabel->setText("Connected...waiting for response");
   tcpSocket.Send( "geometry" );
   tcpSocket.Send( "disconnect" );
   tcpSocket.flush();
}

void MeshDialog::prematureClose(){
   tcpSocket.disconnectFromHost();
   success = false;
   close();
}

void MeshDialog::readData() {

    std::stringstream ss;
    ss << "Conn: " << connN << "(" << data->conn.size() << ")" << " Vert: " << vertN << "(" << data->vert.size() << ")";
    statusLabel->setText(ss.str().c_str());

    if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
    if(connN < 0){
        connN = tcpSocket.RecvInt( );
    }

    while( data->conn.size() < connN ){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        data->conn.push_back( tcpSocket.RecvInt( ) );
    }

    if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
    if(vertN < 0){
        vertN = tcpSocket.RecvInt( );
    }

    while( data->vert.size() < vertN ){
        if( tcpSocket.bytesAvailable() < sizeof(SCI::Vex3) ){ return; }
        data->vert.push_back( tcpSocket.RecvVex3( ) );
    }

    QTimer::singleShot(500, this, SLOT(close()));

    success = true;

}

bool MeshDialog::isSuccessful(){
   return success;
}


void MeshDialog::displayError(QTcpSocket::SocketError socketError) {

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

    close();
}


*/
