#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT       += core gui opengl network
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets


PROJECT(heart)



LIBS += -L../../bin/libs
LIBS += -lcommon
LIBS += -ldata

# glew for windows
win32:LIBS += -lglew32



SET(DESTDIR     ../../bin)
SET(OBJECTS_DIR ../../bin/build/heart_app/.obj)
SET(MOC_DIR     ../../bin/build/heart_app/.moc)
SET(RCC_DIR     ../../bin/build/heart_app/.rcc)
SET(UI_DIR      ../../bin/build/heart_app/.ui)

win32:QMAKE_CXXFLAGS += -openmp
unix:QMAKE_CXXFLAGS  += -fopenmp

unix:LIBS += -lgomp

win32:DEFINES += _CRT_SECURE_NO_WARNINGS

INCLUDEPATH += ../../include
INCLUDEPATH += ../../../Common/include
INCLUDEPATH += .
INCLUDEPATH += ../

SET(#CUDA_SOURCES heart/distance_field.cu heart/voxel_associativity.cu heart/cuda_common.cu)

#/NODEFAULTLIB:libcmt
win32:LIBS += "/nodefaultlib:libcmt"

SOURCES += \
    ../../src/muView/HeartMainWindow.cpp \
    ../../src/muView/HeartMain.cpp \
    ../../src/muView/Histogram.cpp \
    ../../src/muView/VolumeRenderer.cpp \
    ../../src/muView/VoxelAssociativity.cpp \
    ../../src/muView/RenderEngine.cpp \
    ../../src/muView/ParallelCoordinates.cpp \
    ../../src/muView/HeartDock.cpp \
    ../../src/muView/RenderEngine3D.cpp \
    ../../src/muView/RenderEngine2D.cpp \
    ../../src/muView/HeartMainWidget.cpp

HEADERS  += \
    ../../include/muView/HeartMainWindow.h \
    ../../include/muView/Histogram.h \
    ../../include/muView/VolumeRenderer.h \
    ../../include/muView/VoxelAssociativity.h \
    ../../include/muView/RenderEngine.h \
    ../../include/muView/ParallelCoordinates.h \
    ../../include/muView/HeartDock.h \
    ../../include/muView/RenderEngine3D.h \
    ../../include/muView/RenderEngine2D.h \
    ../../include/muView/HeartMainWidget.h


RESOURCES += \
    ../../../Common/fonts/arial.qrc

ADD_EXECUTABLE(heart heart_sources)
