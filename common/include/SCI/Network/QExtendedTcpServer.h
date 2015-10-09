#ifndef QEXTENDEDTCPSERVER_H
#define QEXTENDEDTCPSERVER_H

#include <QTcpServer>

namespace SCI {
    namespace Network {

        class QExtendedTcpServer : public QTcpServer
        {
            Q_OBJECT

        public:
            QExtendedTcpServer(QObject *parent = 0);

            std::string GetIPAddress();
            short       GetPort();


        };

    }
}

#endif // QEXTENDEDTCPSERVER_H
