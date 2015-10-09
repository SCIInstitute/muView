#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

#QT       += core gui opengl
QT += network
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TEMPLATE = lib
CONFIG += staticlib
TARGET = common

DESTDIR     = ../../bin/libs
OBJECTS_DIR = ../../bin/build/common_lib/.obj
MOC_DIR     = ../../bin/build/common_lib/.moc
RCC_DIR     = ../../bin/build/common_lib/.rcc
UI_DIR      = ../../bin/build/common_lib/.ui

INCLUDEPATH += ../../include
INCLUDEPATH += ../../../Common/include

win32:DEFINES += _CRT_SECURE_NO_WARNINGS

SOURCES += \
    ../../../Common/src/GL/oglTextureRECT.cpp \
    ../../../Common/src/GL/oglTexture3D.cpp \
    ../../../Common/src/GL/oglTexture2D.cpp \
    ../../../Common/src/GL/oglTexture1D.cpp \
    ../../../Common/src/GL/oglTexture.cpp \
    ../../../Common/src/GL/oglFrameBufferObject.cpp \
    ../../../Common/src/GL/oglCommon.cpp \
    ../../../Common/src/GL/oglShader.cpp \
    ../../../Common/src/SCI/Timer.cpp \
    ../../../Common/src/SCI/Mat4.cpp \
    ../../../Common/src/SCI/Subset.cpp \
    ../../../Common/src/SCI/Graphics/MarchingCubes.cpp \
    ../../../Common/src/SCI/Camera/ThirdPersonCameraControls.cpp \
    ../../../Common/src/SCI/Camera/OrthoProjection.cpp \
    ../../../Common/src/SCI/Camera/LookAt.cpp \
    ../../../Common/src/SCI/Camera/FrustumProjection.cpp \
    ../../../Common/src/SCI/Camera/FirstPersonCameraControls.cpp \
    ../../../Common/src/SCI/Camera/CameraControls.cpp \
    ../../../Common/src/SCI/Colormap.cpp \
    ../../../Common/src/GL/oglFont.cpp \
    ../../../Common/src/QT/QExtendedMainWindow.cpp \
    ../../../Common/src/QT/QControlWidget.cpp \
    ../../../Common/src/QT/QExtendedVerticalSlider.cpp \
    ../../../Common/src/QT/QDoubleSlider.cpp \
    ../../../Common/src/SCI/SimpleHistogram.cpp

#    ../../../Common/src/SCI/Network/QExtendedTcpSocket.cpp \

unix:SOURCES += ../../../Common/src/GL/glew.c

HEADERS  += \
    ../../../Common/include/SCI/Vex4.h \
    ../../../Common/include/SCI/Vex3.h \
    ../../../Common/include/SCI/Vex2.h \
    ../../../Common/include/SCI/Mat4.h \
    ../../../Common/include/SCI/Timer.h \
    ../../../Common/include/SCI/Network/QExtendedTcpSocket.h \
    ../../../Common/include/SCI/Subset.h \
    ../../../Common/include/SCI/Network/QExtendedTcpServer.h \
    ../../../Common/include/SCI/Utility.h \
    ../../../Common/include/SCI/Graphics/MarchingCubes.h \
    ../../../Common/include/SCI/BoundingBox.h \
    ../../../Common/include/GL/oglTextureRECT.h \
    ../../../Common/include/GL/oglTexture3D.h \
    ../../../Common/include/GL/oglTexture2D.h \
    ../../../Common/include/GL/oglTexture1D.h \
    ../../../Common/include/GL/oglTexture.h \
    ../../../Common/include/GL/oglFrameBufferObject.h \
    ../../../Common/include/GL/oglCommon.h \
    ../../../Common/include/GL/oglShader.h \
    ../../../Common/include/SCI/Camera/ThirdPersonCameraControls.h \
    ../../../Common/include/SCI/Camera/Projection.h \
    ../../../Common/include/SCI/Camera/OrthoProjection.h \
    ../../../Common/include/SCI/Camera/LookAt.h \
    ../../../Common/include/SCI/Camera/FrustumProjection.h \
    ../../../Common/include/SCI/Camera/FirstPersonCameraControls.h \
    ../../../Common/include/SCI/Camera/CameraControls.h \
    ../../../Common/include/SCI/Colormap.h \
    ../../../Common/include/GL/oglFont.h \
    ../../../Common/include/SCI/VexN.h \
    ../../../Common/include/QT/QControlWidget.h \
    ../../../Common/include/QT/QExtendedMainWindow.h \
    ../../../Common/include/SCI/Geometry/CompGeom.h \
    ../../../Common/include/GL/glext.h \
    ../../../Common/include/QT/QExtendedVerticalSlider.h \
    ../../../Common/include/QT/QDoubleSlider.h \
    ../../../Common/include/SCI/SimpleHistogram.h




