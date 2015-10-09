#include <Server/DataThread.h>
#include <Server/DataServer.h>

#include <iostream>

#include <SCI/Vex2.h>

#include <Data/MappedData.h>
#include <Data/VolData.h>
#include <Data/MeshData.h>
#include <Data/PointData.h>

#include <DimensionalityReduction.h>

DataThread::DataThread(int _sid, int _tid, std::vector<DataServer*> * _servers, Data::BasicData * _data, int socketDescriptor, QObject *parent) : QThread(parent), data(_data), data_out(0), servers(_servers), sid(_sid), tid(_tid) {

    connect(  this,      SIGNAL(finished()),                           this, SLOT(deleteLater()));
    connect( &tcpSocket, SIGNAL(SocketError(QTcpSocket::SocketError)), this, SLOT(SocketErrorHandle(QTcpSocket::SocketError)));

    tcpSocket.setSocketDescriptor(socketDescriptor);

}

DataThread::~DataThread(){

    if(data_out) delete data_out;
    std::cout << "DataThread (" << sid << "x" << tid << "): Deleted" << std::endl; fflush(stdout);


}


void DataThread::Disconnect( ){

    tcpSocket.disconnectFromHost();
    if( tcpSocket.state() != QAbstractSocket::UnconnectedState ){
        tcpSocket.waitForDisconnected();
    }
    std::cout << "DataThread (" << sid << "x" << tid << "): Disconnected" << std::endl; fflush(stdout);

}

void DataThread::SocketErrorHandle(QTcpSocket::SocketError socketError){

    std::cout << "DataThread (" << sid << "x" << tid << "): Unknown socket error " << socketError << std::endl; fflush(stdout);

}


void DataThread::run() {

    while( tcpSocket.isConnected() ){
        // check to see if data is waiting
        if( tcpSocket.bytesAvailable() == 0 ){
            SleepMsec(50);
            continue;
        }

        // get the command
        std::string cmd = tcpSocket.RecvString( true );

        // respond to the command
        std::cout << "DataThread (" << sid << "x" << tid << "): Got command '" << cmd << "'" << std::endl; fflush(stdout);
        if( cmd.compare("metadata")   == 0 ){ ProcessMetadata(); continue; }
        if( cmd.compare("servers")    == 0 ){ ProcessServers();  continue; }
        if( cmd.compare("dr_methods") == 0 ){ ProcessMethods();  continue; }
        if( cmd.compare("dr")         == 0 ){ ProcessDimRed();   continue; }
        if( cmd.compare("geometry")   == 0 ){ ProcessGeometry(); continue; }
        if( cmd.compare("disconnect") == 0 ){ break; }
        std::cout << "DataThread (" << sid << "x" << tid << "): Unknown command '" << cmd << "'" << std::endl; fflush(stdout);
    }

    Disconnect();

}

void DataThread::ProcessMethods(){

    tcpSocket.Send( dr.GetMethodCount() );
    for(int i = 0; i < dr.GetMethodCount(); i++){
        tcpSocket.Send( dr.GetMethodName(i) );
    }

}

void DataThread::ProcessGeometry(){

    Data::MeshData*  mdata = dynamic_cast<Data::MeshData*>(data);
    Data::PointData* pdata = dynamic_cast<Data::PointData*>(data);

    int connN = 0;
    void * conn = 0;
    int vertN = 0;
    void * vert = 0;

    if( mdata ){
        connN = mdata->GetConnectivity().size();
        conn  = &(mdata->GetConnectivity()[0]);
        vertN = mdata->GetVertices().size();
        vert  = &(mdata->GetVertices()[0]);
    }
    if( pdata ){
        vertN = pdata->GetVertices().size();
        vert  = &(pdata->GetVertices()[0]);
    }

    tcpSocket.Send( connN );
    if(conn) tcpSocket.Send( conn, sizeof(int)*connN );
    tcpSocket.Send( vertN );
    if(vert) tcpSocket.Send( vert, sizeof(SCI::Vex3)*vertN );

}

void DataThread::ProcessServers(){
    tcpSocket.Send( (int)servers->size() );
    for(int i = 0; i < (int)servers->size(); i++){
        tcpSocket.Send( servers->at(i)->GetName() );
        tcpSocket.Send( servers->at(i)->GetType() );
        //tcpSocket.Send( servers->at(i)->GetIPAddress() );
        tcpSocket.Send( "127.0.0.1" );
        tcpSocket.Send( servers->at(i)->GetPort() );
    }
}

void DataThread::ProcessMetadata(){
    tcpSocket.Send( data->GetDim() );
    tcpSocket.Send( data->GetX() );
    tcpSocket.Send( data->GetY() );
    tcpSocket.Send( data->GetZ() );
}

void DataThread::ProcessDimRed(){
    int method = tcpSocket.RecvInt(true);

    int featN  = tcpSocket.RecvInt(true);
    for(int i = 0; i < featN; i++){
        features.push_back( tcpSocket.RecvInt(true) );
    }

    data_out = new Data::MappedData(data->GetX(),data->GetY(),data->GetZ(),2);
    dr.Start( method, *data, features, *data_out, elements );

    int elemN  = tcpSocket.RecvInt(true);
    if(elemN == -1){
        elemN = data->GetVoxelCount();
        elements = SCI::Subset(data->GetVoxelCount());
    }
    else{
        for(int i = 0; i < elemN; i++){
            elements.push_back( tcpSocket.RecvInt(true) );
        }
    }

    dr.Restart( elements );

    std::cout << "DataThread (" << sid << "x" << tid << "): Dim Red " << method << std::endl; fflush(stdout);
    std::cout << "DataThread (" << sid << "x" << tid << "): Features " << featN << std::endl; fflush(stdout);
    std::cout << "DataThread (" << sid << "x" << tid << "): Elements " << elemN << std::endl; fflush(stdout);

    SCI::Vex2 p;
    for(int i = 0; i < dr.GetFeatureCount(); i++ ){
        dr.GetFeaturePoint( i, p.data );
        tcpSocket.Send( (char*)p.data, sizeof(float)*2 );
    }
    tcpSocket.waitForBytesWritten( );

    int e_sent = 0;
    do {
        while( tcpSocket.isConnected() && e_sent < dr.GetElementCount() ){
            dr.GetElementPoint( e_sent++, p.data );
            tcpSocket.Send( (char*)p.data, sizeof(SCI::Vex2) );
            if( tcpSocket.bytesToWrite() >= 8192 ){
                if( !tcpSocket.waitForBytesWritten( ) ) break;
            }
        }

    } while( tcpSocket.isConnected() && e_sent < dr.GetTotalElementCount() );

    dr.Stop();

}


