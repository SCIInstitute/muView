#ifndef MESHDIALOG_H
#define MESHDIALOG_H

/*
#include <QDialog>
#include <QTcpSocket>
#include <QGridLayout>

//#include <Server/BasicServerThread.h>
#include <SCI/Network/QExtendedTcpSocket.h>

#include <Data/ProxyData.h>

 class QDialogButtonBox;
 class QLabel;
 class QLineEdit;
 class QPushButton;
 class QTcpSocket;

 namespace Dialog {
     class MeshDialog : public QDialog
     {
         Q_OBJECT

     public:
         MeshDialog(Data::ProxyData & data, QWidget *parent = 0);

         bool isSuccessful();

     protected:
         bool success;

     private slots:
         void sendCommand();
         void readData();
         void displayError(QTcpSocket::SocketError socketError);
         void checkForError();

         void prematureClose( );

     private:
         Data::ProxyData * data;
         int connN;
         int vertN;


         QLabel *statusLabel;

         QPushButton *quitButton;

         QDialogButtonBox *buttonBox;
         QGridLayout *mainLayout;

         SCI::Network::QExtendedTcpSocket tcpSocket;

     };

 }

*/

#endif // MESHDIALOG_H
