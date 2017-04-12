#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

QT += core gui opengl network
QT += widgets charts

TEMPLATE = app

TARGET = muView


DESTDIR     = ../../bin
OBJECTS_DIR = ../../bin/build/$$TARGET/.obj
MOC_DIR     = ../../bin/build/$$TARGET/.moc
RCC_DIR     = ../../bin/build/$$TARGET/.rcc
UI_DIR      = ../../bin/build/$$TARGET/.ui



LIBS += -L../../bin/libs
LIBS += -lcommon
LIBS += -ldata





QMAKE_LFLAGS += -F /System/Library/Frameworks/Cocoa.framework/
LIBS += -framework Cocoa
QMAKE_LFLAGS += -F /System/Library/Frameworks/CoreVideo.framework/
LIBS += -framework CoreVideo

LIBS += -L/usr/local/lib -lglfw3 -framework OpenGL


# glew for windows
win32 {
    LIBS += -lglew32
    QMAKE_CXXFLAGS += -openmp
    DEFINES += _CRT_SECURE_NO_WARNINGS
    LIBS += "/nodefaultlib:libcmt"
}
unix {
    QMAKE_CXXFLAGS  += -openmp
    #LIBS += -lgomp
    LIBS += -lshogun -L/usr/local/lib/
}


INCLUDEPATH += ../../include
INCLUDEPATH += /usr/local/include
INCLUDEPATH += ../../../Common/include
#INCLUDEPATH += ../../../Common/drl/drl
INCLUDEPATH += .
INCLUDEPATH += ../


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
    ../../src/Shogun_Wrapper.cpp \
    ../../src/DimensionalityReductionThreaded.cpp \
    ../../src/DimensionalityReduction.cpp \
    ../../src/muView/PCAView.cpp \
    ../../src/muView/ChartCloud.cpp \
    ../../src/muView/ChartRect.cpp \
    ../../src/muView/PCAaxis.cpp


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
    ../../include/Shogun_Wrapper.h \
    ../../include/DimensionalityReductionThreaded.h \
    ../../include/DimensionalityReduction.h \
    ../../include/muView/PCAView.h \
    ../../include/muView/ChartCloud.h \
    ../../include/muView/ChartRect.h \
    ../../include/muView/PCAaxis.h


RESOURCES += \
    ../../../Common/fonts/arial.qrc

