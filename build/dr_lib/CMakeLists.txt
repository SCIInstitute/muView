#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

#QT       += core gui opengl

CONFIG += staticlib
PROJECT(dr)

SET(DESTDIR     ../../bin/libs)
SET(OBJECTS_DIR ../../bin/build/dr_lib/.obj)
SET(MOC_DIR     ../../bin/build/dr_lib/.moc)
SET(RCC_DIR     ../../bin/build/dr_lib/.rcc)
SET(UI_DIR      ../../bin/build/dr_lib/.ui)

INCLUDEPATH += ../../include
INCLUDEPATH += ../../drl/drl

INCLUDEPATH += ../shogun_lib/src

unix {
    # Include OpenMP
    QMAKE_CXXFLAGS += -fopenmp -DYA_OMP

    # Include BLAS
    QMAKE_CXXFLAGS += -DYA_BLAS -DYA_BLASMULT

    #Include LAPACK
    QMAKE_CXXFLAGS += -DYA_LAPACK
}

SOURCES += \
    ../../src/DRL_Wrapper.cpp \

unix:SOURCES += \
    ../../src/Shogun_Wrapper.cpp

HEADERS  += \
    ../../include/DRL_Wrapper.h \

unix:HEADERS  += \
    ../../include/Shogun_Wrapper.h


ADD_LIBRARY(dr ${dr_sources})
