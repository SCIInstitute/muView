s#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT       += core gui opengl network
QT += widgets

TEMPLATE = app

TARGET = heart


LIBS += -lteem -L/usr/local/Cellar/teem/1.11.0/lib
LIBS += -L../../bin/libs
LIBS += -lcommon
LIBS += -ldata

# glew for windows
win32:LIBS += -lglew32
unix {
    LIBS += -lshogun -L/usr/local/lib/
}


DESTDIR     = ../../bin
OBJECTS_DIR = ../../bin/build/heart_app/.obj
MOC_DIR     = ../../bin/build/heart_app/.moc
RCC_DIR     = ../../bin/build/heart_app/.rcc
UI_DIR      = ../../bin/build/heart_app/.ui

win32:QMAKE_CXXFLAGS += -openmp
unix:QMAKE_CXXFLAGS  += -openmp

#unix:LIBS += -lgomp

win32:DEFINES += _CRT_SECURE_NO_WARNINGS

INCLUDEPATH += ../../include
INCLUDEPATH += ../../../Common/include
INCLUDEPATH += .
INCLUDEPATH += ../
INCLUDEPATH += ../../src
INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/local/Cellar/teem/1.11.0/include
#CUDA_SOURCES = heart/distance_field.cu heart/voxel_associativity.cu heart/cuda_common.cu

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
    ../../src/MainWidget.cpp \
    ../../src/DimensionalityReduction.cpp \
    ../../src/Shogun_Wrapper.cpp \
    ../../src/Drawable/DimRedDisplay.cpp \
    ../../src/Drawable/VolumeDisplay.cpp \
    ../../src/Drawable/DrawableObject.cpp \
    ../../src/Drawable/Provenance.cpp \
    ../../src/Drawable/MeshDisplay.cpp \
    ../../src/Drawable/PointDisplay.cpp \
    ../../src/Drawable/ProvenanceNode.cpp \
    ../../src/Drawable/Barchart.cpp \
    ../../src/Drawable/ColorLine.cpp \
    ../../src/Dialog/Metadata.cpp \
    ../../src/Histogram2D.cpp \
    ../../src/BoundingRegion.cpp \
    ../../src/RemoteDimensionalityReduction.cpp \
    ../../../Common/src/SCI/Network/QExtendedTcpSocket.cpp\
    ../../../Common/src/MyLib/PolylineSimplification.cpp\
    ../../../Common/src/Data/BasicData.cpp\
    ../../../Common/src/Data/MappedData.cpp\
    ../../../Common/src/Data/ProxyData.cpp\
    ../../src/muView/PCAView.cpp

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
    ../../include/MainWidget.h \
    ../../include/DimensionalityReduction.h \
    ../../include/Shogun_Wrapper.h \
    ../../include/Drawable/DimRedDisplay.h \
    ../../include/Drawable/VolumeDisplay.h \
    ../../include/Drawable/DrawableObject.h \
    ../../include/Drawable/Provenance.h \
    ../../include/Drawable/MeshDisplay.h \
    ../../include/Drawable/PointDisplay.h \
    ../../include/Drawable/ProvenanceNode.h \
    ../../include/Drawable/Barchart.h \
    ../../include/Drawable/ColorLine.h \
    ../../include/Dialog/Metadata.h \
    ../../include/Histogram2D.h \
    ../../include/BoundingRegion.h \
    ../../include/RemoteDimensionalityReduction.h \
    ../../../Common/include/SCI/Network/QExtendedTcpSocket.h\
    ../../../Common/include/MyLib/PolylineSimplification.h\
    ../../../Common/include/Data/BasicData.h\
    ../../../Common/include/Data/MappedData.h\
    ../../../Common/include/Data/ProxyData.h\
    ../../include/muView/PCAView.h


RESOURCES += \
    ../../../Common/fonts/arial.qrc

