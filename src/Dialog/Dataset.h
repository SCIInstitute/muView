#ifndef DIALOG_DATASET_H
#define DIALOG_DATASET_H

#include <QDialog>
#include <QGridLayout>
#include <QRadioButton>
#include <QGroupBox>
#include <QDialogButtonBox>

namespace Dialog {
    class Dataset : public QDialog {
        Q_OBJECT

    public:
        Dataset(std::vector<std::string> & dataset_list, QWidget *parent = 0);

        int GetSelected( );

    public slots:
        void selectDataset( );

    private:

        std::vector<QRadioButton*> datasetButtons;

        int selected;

        QGroupBox         *groupBox;
        QVBoxLayout       *vbox;
        QPushButton       *selectButton;
        QPushButton       *quitButton;
        QDialogButtonBox  *buttonBox;
        QGridLayout       *mainLayout;

    };
}

#endif
