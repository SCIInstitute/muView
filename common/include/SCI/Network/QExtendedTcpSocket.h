#ifndef SCI_NETWORK_QEXTENDEDTCPSOCKET_H
#define SCI_NETWORK_QEXTENDEDTCPSOCKET_H

#include <QTcpSocket>
#include <QThread>

#include <SCI/Vex2.h>
#include <SCI/Vex3.h>
#include <SCI/Vex4.h>


#if defined(WIN32)
    inline void SleepMsec(int amnt){ Sleep(amnt); }
#elif defined(linux)
    #include <unistd.h>
    inline void SleepMsec(int amnt){ usleep(amnt*1000); }
#endif


namespace SCI {
    namespace Network {
        class QExtendedTcpSocket : public QTcpSocket {
            Q_OBJECT

        public:

            static std::string GetIPAddress();

            QExtendedTcpSocket( QObject * parent = 0 );

            bool isConnected();

            void connectToHost( const char *      addr, unsigned int port, QIODevice::OpenMode openmode = QIODevice::ReadWrite );
            void connectToHost( const std::string addr, unsigned int port, QIODevice::OpenMode openmode = QIODevice::ReadWrite );
            void connectToHost( const QString     addr, unsigned int port, QIODevice::OpenMode openmode = QIODevice::ReadWrite );

            bool SetSocketDescriptor( int socketDescriptor );

            bool Send( const char * str );
            bool Send( std::string str );
            bool Send( int i );
            bool Send( short s );
            bool Send( char c );
            bool Send( float f );
            bool Send( Vex2 v );
            bool Send( Vex3 v );
            bool Send( Vex4 v );
            bool Send( const void * data, int len );

            std::string RecvString( bool block = false );
            int         RecvInt(    bool block = false );
            short       RecvShort(  bool block = false );
            short       RecvChar(   bool block = false );
            float       RecvFloat(  bool block = false );
            Vex2        RecvVex2(   bool block = false );
            Vex3        RecvVex3(   bool block = false );
            Vex4        RecvVex4(   bool block = false );
            bool        RecvBytes(  char * space, int len, bool block = false );

        signals:
            void SocketError(QTcpSocket::SocketError socketError);

        };
    }
}


#endif // SCI_NETWORK_QEXTENDEDTCPSOCKET_H
