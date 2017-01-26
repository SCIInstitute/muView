#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT       += core gui opengl
QT += widgets

TEMPLATE = app

TARGET = dfield

LIBS += -L../../bin/libs


LIBS += -lcommon
LIBS += -ldata

win32 {
    DEFINES += _CRT_SECURE_NO_WARNINGS
}


win32:QMAKE_CXXFLAGS += -openmp
unix:QMAKE_CXXFLAGS  += -fopenmp

#unix:LIBS += -lgomp


DESTDIR     = ../../bin
OBJECTS_DIR = ../../bin/build/dfield_app/.obj
MOC_DIR     = ../../bin/build/dfield_app/.moc
RCC_DIR     = ../../bin/build/dfield_app/.rcc
UI_DIR      = ../../bin/build/dfield_app/.ui

INCLUDEPATH += ../../include
INCLUDEPATH += ../../../Common/include
INCLUDEPATH += ./

SOURCES += \
    ../../src/dfield/dfield_main.cpp \

HEADERS  += \











