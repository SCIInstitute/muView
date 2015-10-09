#include <QtGui/QApplication>

#include <time.h>

#include <Server/DataServer.h>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QCoreApplication::setOrganizationName("SCI Institute");
    QCoreApplication::setOrganizationDomain("Uncertainty");
    QCoreApplication::setApplicationName("HD Uncertainty Server");

    srand( time( 0 ) );

    int cur_port = 3000;

    DataServer is( QHostAddress::Any, cur_port++ );

    is.RegisterDataset( new DataServer( std::string("Fuzzy Segmentation - Lesioned"), std::string("../data/lesionFuzzyALL.nrrd"), std::string("volume"),     QHostAddress::Any, cur_port++ ) );
    is.RegisterDataset( new DataServer( std::string("Fuzzy Segmentation - Normal"),   std::string("../data/normalFuzzyALL.nrrd"), std::string("volume"),     QHostAddress::Any, cur_port++ ) );
    is.RegisterDataset( new DataServer( std::string("Torso"),                         std::string("../data/torso/torso.mesh"),    std::string("mesh"),       QHostAddress::Any, cur_port++ ) );
    is.RegisterDataset( new DataServer( std::string("Heart Conductivity"),            std::string("../data/conductivity.point"),  std::string("pointcloud"), QHostAddress::Any, cur_port++ ) );

    return app.exec();
}
