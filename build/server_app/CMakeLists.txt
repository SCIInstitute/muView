#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT       += core gui network


PROJECT(uncertainty_server)

LIBS += -lcommon -L../../bin/libs


LIBS += -lcommon
LIBS += -ldata
LIBS += -ldr
LIBS += -lteem

unix {
    LIBS += -lshogun
    LIBS += -lgomp
    LIBS += -lblas
    LIBS += -llapack
    LIBS += -lbz2
}

win32 {
    DEFINES += _CRT_SECURE_NO_WARNINGS
}

SET(DESTDIR     ../../bin)
SET(OBJECTS_DIR ../../bin/build/server_app/.obj)
SET(MOC_DIR     ../../bin/build/server_app/.moc)
SET(RCC_DIR     ../../bin/build/server_app/.rcc)
SET(UI_DIR      ../../bin/build/server_app/.ui)

INCLUDEPATH += ../../include
INCLUDEPATH += ../../shogun_lib/src
INCLUDEPATH += ../../shogun_lib/src/shogun

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











ADD_EXECUTABLE(uncertainty_server uncertainty_server_sources)
