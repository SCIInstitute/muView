#include <SCI/Network/QExtendedTcpServer.h>

#include <QNetworkInterface>
#include <iostream>

using namespace SCI::Network;

QExtendedTcpServer::QExtendedTcpServer(QObject * parent) : QTcpServer(parent) { }

std::string QExtendedTcpServer::GetIPAddress(){
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

short QExtendedTcpServer::GetPort(){
    return serverPort();
}


