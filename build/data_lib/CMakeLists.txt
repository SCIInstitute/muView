#-------------------------------------------------
#
# Project created by QtCreator 2011-12-26T20:42:08
#
#-------------------------------------------------

#QT       += core gui opengl

CONFIG += staticlib
PROJECT(data)

SET(DESTDIR     ../../bin/libs)
SET(OBJECTS_DIR ../../bin/build/data_lib/.obj)
SET(MOC_DIR     ../../bin/build/data_lib/.moc)
SET(RCC_DIR     ../../bin/build/data_lib/.rcc)
SET(UI_DIR      ../../bin/build/data_lib/.ui)

INCLUDEPATH += ../../../Common/include

win32:DEFINES += _CRT_SECURE_NO_WARNINGS

win32:QMAKE_CXXFLAGS += -openmp
unix:QMAKE_CXXFLAGS  += -fopenmp

SOURCES += \
    ../../../Common/src/Data/PointData.cpp \
    ../../../Common/src/Data/Mesh/TetMesh.cpp \
    ../../../Common/src/Data/Mesh/TriMesh.cpp \
    ../../../Common/src/Data/Mesh/PointMesh.cpp \
    ../../../Common/src/Data/Mesh/HexMesh.cpp \
    ../../../Common/src/Data/Mesh/QuadMesh.cpp \
    ../../../Common/src/Data/Mesh/KDTree.cpp \
    ../../../Common/src/Data/DistanceField.cpp \
    ../../../Common/src/Data/Mesh/SolidMesh.cpp \
    ../../../Common/src/Data/Mesh/EdgeMesh.cpp \
    ../../../Common/src/Data/MultiDimensionalData.cpp \
    ../../../Common/src/Data/DenseMultiDimensionalData.cpp \
    ../../../Common/src/Data/VolumetricData.cpp \
    ../../../Common/src/Data/Mesh/ColorMesh.cpp \
    ../../../Common/src/Data/Mesh/OrientationMesh.cpp \
    ../../../Common/src/Data/KMeansClustering.cpp \
    ../../../Common/src/Data/MultiDimensionalDataVector.cpp \
    ../../../Common/src/Data/FiberDirectionData.cpp


HEADERS  += \
    ../../../Common/include/Data/PointData.h \
    ../../../Common/include/Data/Mesh/TetMesh.h \
    ../../../Common/include/Data/Mesh/TriMesh.h \
    ../../../Common/include/Data/Mesh/PointMesh.h \
    ../../../Common/include/Data/Mesh/HexMesh.h \
    ../../../Common/include/Data/Mesh/SolidMesh.h \
    ../../../Common/include/Data/Mesh/QuadMesh.h \
    ../../../Common/include/Data/Mesh/KDTree.h \
    ../../../Common/include/Data/FiberDirectionData.h \
    ../../../Common/include/Data/DistanceField.h \
    ../../../Common/include/Data/Mesh/EdgeMesh.h \
    ../../../Common/include/Data/MultiDimensionalData.h \
    ../../../Common/include/Data/DenseMultiDimensionalData.h \
    ../../../Common/include/Data/VolumetricData.h \
    ../../../Common/include/Data/Mesh/ColorMesh.h \
    ../../../Common/include/Data/Mesh/Primitives.h \
    ../../../Common/include/Data/Mesh/OrientationMesh.h \
    ../../../Common/include/Data/KMeansClustering.h \
    ../../../Common/include/Data/MultiDimensionalDataVector.h \
    ../../../Common/include/Data/MappedData.h





ADD_LIBRARY(data ${data_sources})
