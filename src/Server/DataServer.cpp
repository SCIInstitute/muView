#include <Server/DataServer.h>

#include <QtNetwork/QNetworkInterface>
#include <Server/DataThread.h>

#include <QTimer>
#include <iostream>
#include <Data/NrrdData.h>
#include <Data/VolData.h>
#include <Data/MeshData.h>
#include <Data/PointData.h>

DataServer::DataServer( std::string _name, std::string _path, std::string _type, const QHostAddress & address, quint16 port, QObject * parent) : SCI::Network::QExtendedTcpServer(parent), name(_name), path(_path), type(_type), data(0), sid(port), tid(0) {

    if( !listen( address, port ) ){
        printf("initialization failed\n");
    }
    else{
        std::cout << "The server is running on"    << std::endl;
        std::cout << "   IP:   " << GetIPAddress()      << std::endl;
        std::cout << "   Port: " << GetPort()           << std::endl;
    }
    fflush(stdout);

    FILE * infile = fopen(path.c_str(),"rb");
    if(infile){
        fclose(infile);
    }
    else{
        std::cout << "DataServer : ERROR : Could not find " << path << std::endl;
        fflush(stdout);
    }

}

DataServer::DataServer(const QHostAddress & address, quint16 port, QObject *parent ) : SCI::Network::QExtendedTcpServer(parent), type("info"), data(0), sid(port), tid(0) {

    if( !listen( address, port ) ){
        printf("initialization failed\n");
    }
    else{
        std::cout << "The server is running on"    << std::endl;
        std::cout << "   IP:   " << GetIPAddress()      << std::endl;
        std::cout << "   Port: " << GetPort()           << std::endl;
    }
    fflush(stdout);

}

void DataServer::RegisterDataset( DataServer * ds ){
    std::cout << "DataServer (" << sid << "): Dataset registered" << std::endl;
    std::cout << "  " << ds->GetName() << std::endl;
    servers.push_back( ds );
}


std::string DataServer::GetName( ){
    return name;
}

std::string DataServer::GetPath( ){
    return path;
}

std::string DataServer::GetType( ){
    return type;
}

void DataServer::incomingConnection(int socketDescriptor){

    std::cout << "DataServer (" << sid << "): Incoming request" << "\n";    fflush(stdout);

    if( data == 0 && type.compare("volume") == 0 ){
        std::cout << "DataServer : Loading data" << "\n";    fflush(stdout);
        data = new Data::NrrdData( new NrrdVolume<float>(path) );
    }
    if( data == 0 && type.compare("mesh") == 0 ){
        std::cout << "DataServer : Loading data" << "\n";    fflush(stdout);
        data = new Data::MeshData( path.c_str() );
    }
    if( data == 0 && type.compare("pointcloud") == 0 ){
        std::cout << "DataServer : Loading data" << "\n";    fflush(stdout);
        data = new Data::PointData( path.c_str() );
    }

    ( new DataThread( sid, tid++, &servers, data, socketDescriptor, parent() ) )->start();

}
