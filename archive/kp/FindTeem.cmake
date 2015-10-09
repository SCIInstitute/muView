###############################################################
##      FindTeem.cmake
##
## Locate TEEM include paths and libraries
##
## This module defines the following symbols:
##
## TEEM_FOUND           : variable defined, true if teem found
## TEEM_PATH            : path to root teem directory
## TEEM_INCLUDE_PATH    : include directory for teem
## TEEM_LIB_PATH        : lib directory
## TEEM_LIB             : teem library(s)
## TEEM_SHARED          : boolean, use shared lib or not
## TEEM_INCLUDE_FILES   : the .h inlcude files 
##
## Written by Joe Kniss
## Edited by Kristin Potter 9/12/06
###############################################################

###############################################################
## Set the path to the teem directory
###############################################################
SET(TEEM_PATH "/usr/local/" CACHE PATH
    "Path to Gordon Kindlmann's TEEM directory, (teem.sf.net)"
)

###############################################################
## If there is a lib path in the dir, set TEEM_FOUND to 1
###############################################################
IF( EXISTS "${TEEM_PATH}/lib" )
   SET(TEEM_FOUND 1)
ELSE( EXISTS "${TEEM_PATH}/lib" )
   SET(TEEM_FOUND 0)
ENDIF( EXISTS "${TEEM_PATH}/lib" )

###############################################################
## Only look for teem files if the directory is valid
###############################################################
IF( TEEM_FOUND )  

 ###############################################################
 ## Default to shared libraries
 ###############################################################
 SET(TEEM_SHARED 1 
     CACHE BOOL 
     "Use shared libraries for teem, static if false")
 
 ###############################################################
 ## Default to use PNG
 ###############################################################
 SET(TEEM_PNG 1 CACHE BOOL "Use png support in teem")
 
 ###############################################################
 ##  Default to use BZIP
 ############################################################### 
 SET(TEEM_ZIP 1 CACHE BOOL "Use bzip support in teem")

 ###############################################################
 ## Figure out what system we are running on 
 ###############################################################
 SET(TEEM_ARCH_DIR "")

 IF(WIN32)
    SET(TEEM_ARCH_DIR "win32")
 ENDIF(WIN32)

 IF(APPLE)
    SET(TEEM_ARCH_DIR "darwin.32")
 ENDIF(APPLE)

 IF(CYGWIN)
    SET(TEEM_ARCH_DIR "cygwin")
 ENDIF(CYGWIN)

 IF(UNIX)

    IF( ${CMAKE_SYSTEM_NAME} EQUAL "Linux" )
       SET(TEEM_ARCH_DIR "linux.32")
    ENDIF( ${CMAKE_SYSTEM_NAME} EQUAL "Linux" )

    IF( ${CMAKE_SYSTEM_NAME} EQUAL "IRIX" )
       SET(TEEM_ARCH_DIR "irix6.n32")
    ENDIF( ${CMAKE_SYSTEM_NAME} EQUAL "IRIX" )

 ENDIF(UNIX)
 
 ###############################################################
 ##  Find the teem includes
 ###############################################################
 FIND_PATH(TEEM_INCLUDE_PATH "teem/nrrd.h"
            "${TEEM_PATH}/include"
            "${TEEM_PATH}/${TEEM_ARCH_DIR}/include"
 )
 
 SET(TEEM_INCLUDE_FILES "") 
 FILE(GLOB_RECURSE TEEM_INCLUDE_FILES "${TEEM_INCLUDE_PATH}/*.h")

 ###############################################################
 ## Find the teem library
 ###############################################################
 SET(TEEM_LIBPATHS "") 
  
 ## Shared Teem ##
 IF( TEEM_SHARED )
    # any usefull search paths should go here
    SET(TEEM_LIBPATHS
             ${TEEM_PATH}/lib
             ${TEEM_PATH}/lib/shared
             ${TEEM_PATH}/lib/Release 
             ${TEEM_PATH}/${TEEM_ARCH_DIR}/lib
             ${TEEM_PATH}/${TEEM_ARCH_DIR}/lib/shared
             ${TEEM_PATH}/${TEEM_ARCH_DIR}/lib/Release ) 
 ## Static Teem ##
 ELSE( TEEM_SHARED )
    SET(TEEM_LIBPATHS
             ${TEEM_PATH}/lib
             ${TEEM_PATH}/lib/static
             ${TEEM_PATH}/lib/StaticRelease 
             ${TEEM_PATH}/${TEEM_ARCH_DIR}/lib
             ${TEEM_PATH}/${TEEM_ARCH_DIR}/lib/static
             ${TEEM_PATH}/${TEEM_ARCH_DIR}/lib/StaticRelease
    ) 
 ENDIF( TEEM_SHARED )

 FIND_LIBRARY(TEEM_LIB 
           NAMES teem teem.lib libteem.so libteem.a
           PATHS ${TEEM_LIBPATHS} )

 ###############################################################
 #  PNG
 ###############################################################
 IF(TEEM_PNG)
   FIND_LIBRARY(PNG_LIB
        NAMES png png.lib libpng.a libpng.dylib libpng.so
        PATHS
           /usr/lib
           /usr/local/lib
           ${TEEM_LIBPATHS}
   )
 SET(TEEM_LIB ${TEEM_LIB} ${PNG_LIB})
 ENDIF(TEEM_PNG)

 ###############################################################
 #  TEEM ZIP (BZ2 and Z)
 ###############################################################
 IF(TEEM_ZIP)
   FIND_LIBRARY(BZ2_LIB 
     NAMES bz2 bz2.lib libbz2.a libbz2.dylib libbz2.so
     PATHS
       /usr/lib
       /usr/local/lib
       /sw/lib
       ${TEEM_LIBPATHS})
   SET(TEEM_LIB ${TEEM_LIB} ${BZ2_LIB})
   FIND_LIBRARY(Z_LIB 
     NAMES z z.lib libz.a libz.dylib libz.so
     PATHS
       /usr/lib
       /usr/local/lib
       /sw/lib
       ${TEEM_LIBPATHS})
   SET(TEEM_LIB ${TEEM_LIB} ${Z_LIB})
 ENDIF(TEEM_ZIP)

ENDIF( TEEM_FOUND )
