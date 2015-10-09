#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
## Minimum CMake Required to Build and the policy
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
## Set the top-level project and init variables
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
PROJECT(ENTROPY)
SET(PROJ_DIRS "")

# Path for code specific to this project
SET(ENTROPY_PATH ${CMAKE_SOURCE_DIR}) 

# Path for all includes
SET(ENTROPY_INC_DIRS "./")

# Path for all link directories
SET(ENTROPY_LIB_DIRS "")

# All libraries
SET(ENTROPY_LIBS "")

# ALL defines
SET(ENTROPY_DEFS "")

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
## By default, build shared libraries.
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
SET(BUILD_SHARED ON)

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
## Add appropriate debugging flags
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
IF(CMAKE_BUILD_TYPE MATCHES "debug")
  ADD_DEFINITIONS(-D_DEBUG -Wall)
  IF(UNIX)
    ADD_DEFINITIONS(-Wall)
  ENDIF(UNIX)
ENDIF(CMAKE_BUILD_TYPE MATCHES "debug")

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
## Set the Windows defines
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
IF(WIN32)
  SET(ENTROPY_DEFS ${ENTROPY_DEFS} -D_WIN32)
ENDIF(WIN32)

#*************************************************************#
# *** Required Packages:
#*************************************************************#
#- - - - - - - - - - - - - Teem - - - - - - - - - - - - - - - # 
INCLUDE(FindTeem.cmake )
SET(ENTROPY_INC_DIRS ${ENTROPY_INC_DIRS} ${TEEM_INCLUDE_PATH} )
SET(ENTROPY_LIB_DIRS ${ENTROPY_LIB_DIRS} ${TEEM_LIB_PATH} )
SET(ENTROPY_LIBS     ${ENTROPY_LIBS}     ${TEEM_LIB} )

#*************************************************************#
# *** Project Applications:
#*************************************************************#
SET(ENTROPY_INC volume.h)
SET(ENTROPY_SRC )

## Include the files
INCLUDE_DIRECTORIES( ${ENTROPY_INC_DIRS} )

## Create the executable
ADD_EXECUTABLE(calcEntropyVol ${ENTROPY_INC} ${ENTROPY_SRC} calcEntropyVol.cpp) 
TARGET_LINK_LIBRARIES(calcEntropyVol ${ENTROPY_LIBS} )
