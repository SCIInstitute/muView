#include <QtGui/QApplication>
#include <MainWindow.h>
#include <time.h>

/*
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QCoreApplication::setOrganizationName("SCI Institute");
    QCoreApplication::setOrganizationDomain("uncertainty");
    QCoreApplication::setApplicationName("HD Uncertainty");

    srand( time( 0 ) );

    MainWindow w;
    w.show();
    
    return a.exec();
}
*/

#include <Dialog/Connection.h>
#include <Dialog/Dataset.h>
#include <Dialog/Metadata.h>
#include <Dialog/MeshDialog.h>

 int main(int argc, char *argv[])
 {
     QApplication app(argc, argv);

     QCoreApplication::setOrganizationName("SCI Institute");
     QCoreApplication::setOrganizationDomain("uncertainty");
     QCoreApplication::setApplicationName("HD Uncertainty");

     Dialog::Connection client;
     client.show();
     client.exec();
     if( !client.isSuccessful() ){ return 0; }

     Dialog::Dataset dataset(client.dataset_list);
     dataset.show();
     dataset.exec();
     int sel = dataset.GetSelected();

     if( sel == -1 ){ return 0; }

     std::cout << "Dataset selected: " << sel << std::endl;
     fflush(stdout);

     Dialog::Metadata metadata( client.dataset_ip[sel], client.dataset_port[sel] );
     metadata.show();
     metadata.exec();

     if( !metadata.isSuccessful() ){ return 0; }

     //Data::ProxyData * data = new Data::ProxyData( client.dataset_ip[sel].c_str(), client.dataset_port[sel], metadata.GetX(), metadata.GetY(), metadata.GetZ(), metadata.GetDim() );

     //Dialog::MeshDialog meshdialog( *data );
     //meshdialog.show();
     //meshdialog.exec();

     //if( !meshdialog.isSuccessful() ){ return 0; }


     MainWindow w( &metadata );
     w.show();

     return app.exec();
 }
