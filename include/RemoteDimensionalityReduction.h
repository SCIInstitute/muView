#ifndef REMOTEDIMENSIONALITYREDUCTION_H
#define REMOTEDIMENSIONALITYREDUCTION_H


#include <QObject>
#include <Data/BasicData.h>
#include <Data/ProxyData.h>
#include <SCI/Subset.h>
#include <SCI/Network/QExtendedTcpSocket.h>

class RemoteDimensionalityReduction : public QThread {
    Q_OBJECT


protected:
    SCI::Network::QExtendedTcpSocket tcpSocket;

    SCI::Subset features, elements;
    int method;

    Data::BasicData * data_out;
    Data::ProxyData * data_in;

    int featN, elemN;

    SCI::Vex2 min_elem, max_elem;

private slots:
    //void sendCommand( );
    //void recvCommand( );

signals:
    void computationComplete(bool);

protected:

    virtual void run ();

    /*
public:
    static int          GetMethodCount();
    static const char * GetMethodName( int i );
*/

public:
    RemoteDimensionalityReduction( );

    int GetTotalFeatureCount() const ;
    int GetFeatureCount() const;
    void GetFeaturePoint( int elem, float  * pout ) const;
    void GetFeaturePoint( int elem, double * pout ) const;
    int  GetFeatureVoxID( int elem ) const ;

    int GetTotalElementCount() const;
    int GetElementCount() const;
    void GetElementPoint( int elem, float  * pout ) const;
    void GetElementPoint( int elem, double * pout ) const;
    int  GetElementVoxID( int elem ) const ;

    void Start( int method, Data::ProxyData & _data_in, SCI::Subset & _features, Data::BasicData & _data_out, SCI::Subset & _elements );
    void Restart( SCI::Subset & _elements );
    void Stop( );

    float GetMinValue( int dim ) const ;

    float GetMaxValue( int dim ) const ;

    int GetCurrentMethod() const ;

};



#endif // REMOTEDIMENSIONALITYREDUCTION_H
