#ifndef DATASERVER_H
#define DATASERVER_H

#include <Data/BasicData.h>
#include <SCI/Network/QExtendedTcpServer.h>

class DataServer : public SCI::Network::QExtendedTcpServer
{
    Q_OBJECT

public:
    DataServer(std::string name, std::string path, std::string type, const QHostAddress &address = QHostAddress::Any, quint16 port = 0, QObject *parent = 0);
    DataServer(const QHostAddress & address, quint16 port, QObject *parent = 0);

    void RegisterDataset( DataServer * ds );

    std::string GetName( );
    std::string GetPath( );
    std::string GetType( );

protected:

    void incomingConnection(int socketDescriptor);

    std::string name;
    std::string path;
    std::string type;

    Data::BasicData * data;

    std::vector< DataServer* > servers;

    int sid, tid;

};

#endif // DATASERVER_H
