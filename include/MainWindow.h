#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QApplication>

#include <MainWidget.h>

#include <Dialog/Metadata.h>

class MainWindow : public QMainWindow {

    Q_OBJECT
    
public:

    MainWindow( Dialog::Metadata * data, QWidget *parent = 0);
    ~MainWindow();

    virtual QSize minimumSizeHint() const;
    virtual QSize sizeHint() const;

protected:

    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void keyReleaseEvent ( QKeyEvent * event );

protected:

    FILE              * log_file;

    QMenu             * file_menu;
    QAction           * exit;

};



#endif // MAINWINDOW_H
