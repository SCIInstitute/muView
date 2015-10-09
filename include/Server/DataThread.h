#ifndef DATATHREAD_H
#define DATATHREAD_H

#include <QThread>
#include <QtNetwork/QTcpSocket>

#include <SCI/Vex2.h>
#include <SCI/Vex3.h>
#include <SCI/Vex4.h>
#include <SCI/Subset.h>

//#include <Server/BasicServerThread.h>
#include <SCI/Network/QExtendedTcpSocket.h>
#include <Data/BasicData.h>
#include <DimensionalityReduction.h>

class DataServer;

class DataThread : public QThread
{
    Q_OBJECT

public:
    DataThread(int sid, int tid, std::vector<DataServer*> * servers, Data::BasicData * data, int socketDescriptor, QObject *parent = 0);
    virtual ~DataThread();

protected:

    void Disconnect( );

    void ProcessMetadata();
    void ProcessDimRed();
    void ProcessServers();
    void ProcessGeometry();
    void ProcessMethods();

    virtual void run();

protected slots:

    void SocketErrorHandle(QTcpSocket::SocketError socketError);

protected:
    SCI::Network::QExtendedTcpSocket tcpSocket;

    std::vector<DataServer*> * servers;

    DimensionalityReduction dr;

    Data::BasicData * data;
    Data::BasicData * data_out;

    SCI::Subset features;
    SCI::Subset elements;

    int tid, sid;

};

#endif // DATATHREAD_H
