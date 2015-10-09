message(STATUS "downloading...
     src='http://www.softagalleria.net/download/dirent/dirent-1.13.zip'
     dst='D:/Code/Common/shogun-3.0.0/third_party/MSDirent/dirent-1.13.zip'
     timeout='none'")




file(DOWNLOAD
  "http://www.softagalleria.net/download/dirent/dirent-1.13.zip"
  "D:/Code/Common/shogun-3.0.0/third_party/MSDirent/dirent-1.13.zip"
  SHOW_PROGRESS
  EXPECTED_HASH;MD5=4a4fdd27848dde028bd5e9712e2636bc
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'http://www.softagalleria.net/download/dirent/dirent-1.13.zip' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
