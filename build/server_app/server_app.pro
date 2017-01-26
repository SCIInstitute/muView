#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT       += core gui network

TEMPLATE = app

TARGET = uncertainty_server

LIBS += -lcommon -L../../bin/libs


LIBS += -lcommon
LIBS += -ldata
LIBS += -ldr
LIBS += -lteem -L/usr/local/Cellar/teem/1.11.0/lib/

unix {
    LIBS += -lshogun -L/usr/local/lib/
    #LIBS += -lgomp
    LIBS += -lblas
    LIBS += -llapack
    LIBS += -lbz2
}

win32 {
    DEFINES += _CRT_SECURE_NO_WARNINGS
}

DESTDIR     = ../../bin
OBJECTS_DIR = ../../bin/build/server_app/.obj
MOC_DIR     = ../../bin/build/server_app/.moc
RCC_DIR     = ../../bin/build/server_app/.rcc
UI_DIR      = ../../bin/build/server_app/.ui

INCLUDEPATH += ../../include
INCLUDEPATH += /usr/local/include
INCLUDEPATH += ../shogun_lib/src
INCLUDEPATH += ../shogun_lib/src/shogun
INCLUDEPATH += /usr/local/Cellar/teem/1.11.0/include/

SOURCES += \
    ../../src/Server/DataServer.cpp \
    ../../src/Server/DataThread.cpp \
    ../../src/Data/NrrdData.cpp \
    ../../src/Server/server_main.cpp \
    ../../src/DimensionalityReduction.cpp \
    ../../src/DimensionalityReductionThreaded.cpp

HEADERS  += \
    ../../include/Server/DataServer.h \
    ../../include/Server/DataThread.h \
    ../../include/DimensionalityReduction.h \
    ../../include/Data/NrrdData.h \
    ../../include/volume.h \
    ../../include/DimensionalityReductionThreaded.h











