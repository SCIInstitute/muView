set(file "D:/Code/Common/shogun-3.0.0/third_party/MSDirent/dirent-1.13.zip")
message(STATUS "verifying file...
     file='${file}'")
set(expect_value "4a4fdd27848dde028bd5e9712e2636bc")
file(MD5 "${file}" actual_value)
if("${actual_value}" STREQUAL "${expect_value}")
  message(STATUS "verifying file... done")
else()
  message(FATAL_ERROR "error: MD5 hash of
  ${file}
does not match expected value
  expected: ${expect_value}
    actual: ${actual_value}
")
endif()
