#include <QApplication>
#include <muView/HeartMainWindow.h>
#include <time.h>


int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);


    QCoreApplication::setOrganizationName( "SCI Institute" );
    QCoreApplication::setOrganizationDomain( "Uncertainty" );
    QCoreApplication::setApplicationName( "muView" );

    SCI::PrintLicense( "muView : Multifield Uncertainty Viewer", "Paul Rosen", "2013" );

    MainWindow w;
    w.setWindowTitle( QString("muView : Multifield Uncertainty Viewer") );
    w.show();

    return app.exec();
}



