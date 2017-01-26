
 #ifndef DIALOG_METADATA_H
 #define DIALOG_METADATA_H

#include <QDialog>
#include <QLabel>
#include <QDialogButtonBox>
#include <QTcpSocket>
#include <QGridLayout>

//#include <Server/BasicServerThread.h>
#include <SCI/Network/QExtendedTcpSocket.h>
#include <Data/ProxyData.h>

/*
class QDialogButtonBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTcpSocket;
*/
 namespace Dialog {
     class Metadata : public QDialog
     {
         Q_OBJECT

     public:
         Metadata(std::string ip, short port, QWidget *parent = 0);

         bool isSuccessful();

         int GetX();
         int GetY();
         int GetZ();
         int GetDim();

         std::string GetMethod( int i );
         int         GetMethodCount();

         Data::ProxyData * GetProxyData();

     protected:
         bool success;

     private slots:
         void sendCommand();
         void readData();
         void displayError(QTcpSocket::SocketError socketError);
         void checkForError();

         void prematureClose( );

     private:
         int x,y,z,dim;
         int method_cnt;
         int connN;
         int vertN;

         std::string ip;
         unsigned short port;

         std::vector<std::string> methods;

         QLabel *statusLabel;

         QPushButton *quitButton;

         QDialogButtonBox *buttonBox;
         QGridLayout *mainLayout;

         SCI::Network::QExtendedTcpSocket tcpSocket;

         Data::ProxyData * data;


     };

 }

 #endif
