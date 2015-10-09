#include <MainWindow.h>

#include <QMenuBar>

MainWindow::MainWindow(Dialog::Metadata *data, QWidget *parent) : QMainWindow(parent){

    // Set window title
    setWindowTitle(tr("HD Uncertainty"));

    // Setup File Menu
    file_menu = menuBar()->addMenu("&File");
    {
        // Setup Exit Menu
        exit = new QAction("E&xit", this );
        {
            exit->setShortcut(tr("CTRL+X"));
            file_menu->addAction(exit);
            connect(exit, SIGNAL(triggered()), qApp, SLOT(quit()));
        }
    }


    // Create a status bar
    statusBar();

    // open the log file
    log_file = fopen("log.txt","w");

    setCentralWidget( new MainWidget(data, menuBar(),this) );

}

MainWindow::~MainWindow(){
    // close the log file
    if(log_file){ fclose(log_file); }
}

// Minimum window size
QSize MainWindow::minimumSizeHint() const { return QSize(50, 50); }

// Desired window size
QSize MainWindow::sizeHint() const { return QSize(1600, 1200); }

void MainWindow::keyPressEvent ( QKeyEvent * ){ }

void MainWindow::keyReleaseEvent ( QKeyEvent * ){ }

