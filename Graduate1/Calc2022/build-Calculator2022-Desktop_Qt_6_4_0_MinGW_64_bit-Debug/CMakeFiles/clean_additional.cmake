# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\Calculator2022_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\Calculator2022_autogen.dir\\ParseCache.txt"
  "Calculator2022_autogen"
  )
endif()
