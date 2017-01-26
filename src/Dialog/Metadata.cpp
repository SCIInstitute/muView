

 #include <QtGui>
 #include <QtNetwork>

#include <SCI/Vex2.h>

#include <Dialog/Metadata.h>
#include <iostream>
#include <sstream>

#include<QPushButton>
#include<QMessageBox>

using namespace Dialog;

Metadata::Metadata(std::string _ip, short _port, QWidget *parent) : QDialog(parent) {

    ip = _ip;
    port = _port;

    std::cout << "Requesting metadata from " << ip << ":" << port << std::endl;
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

    setWindowTitle(tr("Server Query"));

    connect(  quitButton, SIGNAL(clicked()),   this, SLOT(prematureClose()) );
    connect( &tcpSocket,  SIGNAL(connected()), this, SLOT(sendCommand()) );
    connect( &tcpSocket,  SIGNAL(readyRead()), this, SLOT(readData()) );
    connect( &tcpSocket,  SIGNAL(SocketError(QTcpSocket::SocketError)), this, SLOT(displayError(QTcpSocket::SocketError)));

    dim = -1;
    x = -1;
    y = -1;
    z = -1;
    method_cnt = -1;
    data = 0;
    connN = -1;
    vertN = -1;

    tcpSocket.abort();
    tcpSocket.connectToHost( tr(ip.c_str()), port );
    QTimer::singleShot(30000, this, SLOT(checkForError()));

}


void Metadata::checkForError(){
    if(!success){
        tcpSocket.abort();
        displayError( QTcpSocket::HostNotFoundError );
    }
}

void Metadata::sendCommand(){
    statusLabel->setText("Connected...waiting for response");
    tcpSocket.Send( "metadata" );
    tcpSocket.Send( "dr_methods" );
    tcpSocket.Send( "geometry" );
    tcpSocket.Send( "disconnect" );
    tcpSocket.flush();
}

void Metadata::prematureClose(){
    tcpSocket.disconnectFromHost();
    success = false;
    close();
}

void Metadata::readData() {

    if(dim < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        dim = tcpSocket.RecvInt( );
    }

    if(x < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        x = tcpSocket.RecvInt( );
    }

    if(y < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        y = tcpSocket.RecvInt( );
    }

    if(z < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        z = tcpSocket.RecvInt( );
    }

    if(method_cnt < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        method_cnt = tcpSocket.RecvInt( );
    }

    while( (int)methods.size() < method_cnt){
        if( tcpSocket.bytesAvailable() <= 0 ){ return; }
        methods.push_back( tcpSocket.RecvString( true ) );
    }

    if( data == 0 ){
        data = new Data::ProxyData( ip.c_str(), port, x, y, z, dim );
    }

    if(connN < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        connN = tcpSocket.RecvInt( );
    }

    while( (int)data->conn.size() < connN ){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        data->conn.push_back( tcpSocket.RecvInt( ) );
    }

    if(vertN < 0){
        if( tcpSocket.bytesAvailable() < sizeof(int) ){ return; }
        vertN = tcpSocket.RecvInt( );
    }

    while( (int)data->vert.size() < vertN ){
        if( tcpSocket.bytesAvailable() < sizeof(SCI::Vex3) ){ return; }
        data->vert.push_back( tcpSocket.RecvVex3( ) );
    }

    tcpSocket.disconnectFromHost();

    std::stringstream ss;
    ss << "Dim:  " << dim << std::endl;
    ss << "Size: " << x << " x " << y << " x " << z << std::endl;
    ss << "Methods: " << method_cnt;
    for(int i = 0; i < (int)methods.size(); i++){
        ss << std::endl << "    " << methods[i];
    }
    ss << std::endl << "Connectivity: " << connN << "(" << data->conn.size() << ")";
    ss << std::endl << "Vertices: " << vertN << "(" << data->vert.size() << ")";
    statusLabel->setText(ss.str().c_str());

    QTimer::singleShot(500, this, SLOT(close()));

    success = true;

}

int Metadata::GetX(){   return x;   }
int Metadata::GetY(){   return y;   }
int Metadata::GetZ(){   return z;   }
int Metadata::GetDim(){ return dim; }
std::string Metadata::GetMethod( int i ){ return methods[i]; }
int Metadata::GetMethodCount(){ return methods.size(); }
Data::ProxyData * Metadata::GetProxyData(){ return data; }

bool Metadata::isSuccessful(){
    return success;
}


 void Metadata::displayError(QTcpSocket::SocketError socketError) {

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

