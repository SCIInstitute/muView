#include <SCI/Network/QExtendedTcpSocket.h>

#include <QTcpSocket>
#include <QNetworkInterface>
#include <iostream>

using namespace SCI;
using namespace SCI::Network;

std::string QExtendedTcpSocket::GetIPAddress(){
    // find out which IP to connect to
    QString ipAddress;
    QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();
    // use the first non-localhost IPv4 address
    for (int i = 0; i < ipAddressesList.size(); ++i) {
        if (ipAddressesList.at(i) != QHostAddress::LocalHost &&
            ipAddressesList.at(i).toIPv4Address())
            ipAddress = ipAddressesList.at(i).toString();
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    return std::string( qPrintable(ipAddress) );
}


QExtendedTcpSocket::QExtendedTcpSocket(QObject *parent) : QTcpSocket(parent) { }

bool QExtendedTcpSocket::SetSocketDescriptor( int socketDescriptor ){
    if (!setSocketDescriptor(socketDescriptor) ) {
        emit SocketError(error());
        return false;
    }
    return true;
}

void QExtendedTcpSocket::connectToHost( const char * addr, unsigned int port, QIODevice::OpenMode openmode ){
    QTcpSocket::connectToHost( tr(addr), port, openmode );
}

void QExtendedTcpSocket::connectToHost( const std::string addr, unsigned int port, QIODevice::OpenMode openmode ){
    QTcpSocket::connectToHost( tr(addr.c_str()), port, openmode );
}

void QExtendedTcpSocket::connectToHost( const QString addr, unsigned int port, QIODevice::OpenMode openmode ){
    QTcpSocket::connectToHost( addr, port, openmode );
}



bool QExtendedTcpSocket::Send( const char * str ){
    int len = (int)strlen(str)+1;
    if( !Send( len ) ) return false;
    return write( str, len ) == len;
}

bool QExtendedTcpSocket::isConnected(){
    return state() == QAbstractSocket::ConnectedState;
}

bool QExtendedTcpSocket::Send( std::string str ){
    return Send( str.c_str() );
}

bool QExtendedTcpSocket::Send( int i ){   return write( (char*)&i,       sizeof(int)   ) ==    sizeof(int)   ; }
bool QExtendedTcpSocket::Send( short s ){ return write( (char*)&s,       sizeof(short) ) ==    sizeof(short) ; }
bool QExtendedTcpSocket::Send( char c ){  return write( (char*)&c,       sizeof(char)  ) ==    sizeof(char)  ; }
bool QExtendedTcpSocket::Send( float f ){ return write( (char*)&f,       sizeof(float) ) ==    sizeof(float) ; }
bool QExtendedTcpSocket::Send( Vex2 v ){  return write( (char*)v.data, 2*sizeof(float) ) == (2*sizeof(float)); }
bool QExtendedTcpSocket::Send( Vex3 v ){  return write( (char*)v.data, 3*sizeof(float) ) == (3*sizeof(float)); }
bool QExtendedTcpSocket::Send( Vex4 v ){  return write( (char*)v.data, 4*sizeof(float) ) == (4*sizeof(float)); }
bool QExtendedTcpSocket::Send( const void * data, int len ){ return write( (char*)data, len ) == len; }


std::string QExtendedTcpSocket::RecvString( bool block ){

    // return value
    std::string ret;

    // read the length of the string
    int len = RecvInt( block );

    // Not enough data
    if( len == -1 ) return ret;

    // get the string
    char * buf = new char[len];
    if( RecvBytes( buf, len, block ) ){ ret = std::string(buf); }
    delete [] buf;

    return ret;
}


short QExtendedTcpSocket::RecvChar(bool block){
    char ret;
    if( RecvBytes( (char*)&ret, sizeof(char), block ) ) return ret;
    return -1;
}

short QExtendedTcpSocket::RecvShort(bool block){
    short ret;
    if( RecvBytes( (char*)&ret, sizeof(short), block ) ) return ret;
    return -1;
}

int QExtendedTcpSocket::RecvInt(bool block){
    int ret;
    if( RecvBytes( (char*)&ret, sizeof(int), block ) ) return ret;
    return -1;
}

float QExtendedTcpSocket::RecvFloat(bool block){
    float ret;
    if( RecvBytes( (char*)&ret, sizeof(float), block ) ) return ret;
    return -1;
}

Vex2 QExtendedTcpSocket::RecvVex2(bool block){
    Vex2 ret;
    if( RecvBytes( (char*)&ret, 2*sizeof(float), block ) ) return ret;
    return VEX2_INVALID;
}

Vex3 QExtendedTcpSocket::RecvVex3(bool block){
    Vex3 ret;
    if( RecvBytes( (char*)&ret, 3*sizeof(float), block ) ) return ret;
    return VEX3_INVALID;
}

Vex4 QExtendedTcpSocket::RecvVex4(bool block){
    Vex4 ret;
    if( RecvBytes( (char*)ret.data, 4*sizeof(float), block ) ) return ret;
    return VEX4_INVALID;
}

bool QExtendedTcpSocket::RecvBytes(  char * space, int len, bool block ){


    // Wait if blocking
    while (block && bytesAvailable() < (qint64)len ) SleepMsec(10);

    // Not enough data
    if (bytesAvailable() < (qint64)(len)) return false;

    // Read the data
    read( space, len );

    return true;

}









