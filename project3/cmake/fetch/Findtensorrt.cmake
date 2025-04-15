list(APPEND CMAKE_MESSAGE_INDENT "[trt]  ")

set(TRT_FOUND 1)
set(TRT_INCLUDE_DIR        ${TRT_DIR}/include)
set(TRT_LIBRARY_DIR        ${TRT_DIR}/lib)
set(TRT_LIBRARY            nvinfer nvinfer_plugin nvonnxparser nvparsers)


message("TRT_FOUND:       ${TRT_FOUND}")
message("TRT_INCLUDE_DIR: ${TRT_INCLUDE_DIR}")
message("TRT_LIBRARY_DIR: ${TRT_LIBRARY_DIR}")
message("TRT_LIBRARY:     ${TRT_LIBRARY}")



include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(trt 
  REQUIRED_VARS TRT_INCLUDE_DIR TRT_LIBRARY_DIR  TRT_LIBRARY
)


list(POP_BACK CMAKE_MESSAGE_INDENT)
