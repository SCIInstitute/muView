#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT       += core gui opengl network

TEMPLATE = app

TARGET = uncertainty_client

# Include TEEM library
LIBS += -lteem
LIBS += -lcommon
LIBS += -ldata
LIBS += -ldr
unix{
    LIBS += -lshogun
}

LIBS += -lcommon -L../../bin/libs

# glew for windows
win32:LIBS += -lglew32

unix {
    # Include OpenMP
    LIBS += -lgomp

    # Include BLAS
    LIBS += -lblas

    #Include LAPACK
    LIBS += -llapack

    LIBS += -lbz2
}



DESTDIR     = ../../bin
OBJECTS_DIR = ../../bin/build/uncertainty_client_app/.obj
MOC_DIR     = ../../bin/build/uncertainty_client_app/.moc
RCC_DIR     = ../../bin/build/uncertainty_client_app/.rcc
UI_DIR      = ../../bin/build/uncertainty_client_app/.ui

win32:DEFINES += _CRT_SECURE_NO_WARNINGS

INCLUDEPATH += ../../include
INCLUDEPATH += .
INCLUDEPATH += ../
#INCLUDEPATH += ../drl/drl
INCLUDEPATH += ../shogun_lib/src
INCLUDEPATH += ../shogun_lib/src/shogun

SOURCES += \
    ../../src/MainWindow.cpp \
    ../../src/main.cpp \
    ../../src/MainWidget.cpp \
    ../../src/Data/NrrdData.cpp \
    ../../src/Drawable/DrawableObject.cpp \
    ../../src/Drawable/VolumeDisplay.cpp \
    ../../src/Drawable/DimRedDisplay.cpp \
    ../../src/Drawable/Provenance.cpp \
    ../../src/Drawable/ProvenanceNode.cpp \
    ../../src/Drawable/BarChart.cpp \
    ../../src/Drawable/ColorLine.cpp \
    ../../src/Histogram2D.cpp \
    ../../src/BoundingRegion.cpp \
    ../../src/Drawable/MeshDisplay.cpp \
    ../../src/Drawable/PointDisplay.cpp \
    Dialog/Dataset.cpp \
    Dialog/Connection.cpp \
    Dialog/Metadata.cpp \
    ../../src/Data/ProxyData.cpp \
    ../../src/RemoteDimensionalityReduction.cpp \
    Dialog/MeshDialog.cpp

HEADERS  += \
    ../../include/MainWindow.h \
    ../../include/volume.h \
    ../../include/MainWidget.h \
    ../../include/Data/NrrdData.h \
    ../../include/Drawable/DrawableObject.h \
    ../../include/Drawable/VolumeDisplay.h \
    ../../include/Drawable/DimRedDisplay.h \
    ../../include/Drawable/Provenance.h \
    ../../include/Drawable/ProvenanceNode.h \
    ../../include/Drawable/ColorLine.h \
    ../../include/Histogram2D.h \
    ../../include/BoundingRegion.h \
    ../../include/Drawable/MeshDisplay.h \
    ../../include/Drawable/BarChart.h \
    ../../include/Drawable/PointDisplay.h \
    Dialog/Dataset.h \
    Dialog/Connection.h \
    Dialog/Metadata.h \
    ../../include/Data/ProxyData.h \
    ../../include/RemoteDimensionalityReduction.h \
    Dialog/MeshDialog.h


OTHER_FILES += \
    ../../shaders/vol.vert.glsl \
    ../../shaders/vol.frag.glsl


# Copies the given files to the destination directory
defineTest(copyToDestdir) {
    files = $$1

    for(FILE, files) {
        DDIR = $$DESTDIR

        # Replace slashes in paths with backslashes for Windows
        win32:FILE ~= s,/,\\,g
        win32:DDIR ~= s,/,\\,g

        QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$FILE) $$quote($$DDIR) $$escape_expand(\\n\\t)
    }

    export(QMAKE_POST_LINK)
}

copyToDestdir( $$OTHER_FILES )






