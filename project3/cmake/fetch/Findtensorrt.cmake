include(FindPackageHandleStandardArgs)

list(APPEND CMAKE_MESSAGE_INDENT "[tensorrt]  ")

set(TRT_FOUND 1)
set(TRT_INCLUDE_DIR        ${TRT_DIR}/include)
set(TRT_LIBRARY_DIR        ${TRT_DIR}/lib)
set(TRT_LIBRARY            nvinfer nvinfer_plugin nvonnxparser nvparsers)
set(TRT_SAMPLE_DIR         ${TRT_DIR}/samples)

message("TRT_FOUND:       ${TRT_FOUND}")
message("TRT_INCLUDE_DIR: ${TRT_INCLUDE_DIR}")
message("TRT_LIBRARY_DIR: ${TRT_LIBRARY_DIR}")
message("TRT_LIBRARY:     ${TRT_LIBRARY}")
message("TRT_SAMPLE_DIR:     ${TRT_SAMPLE_DIR}")


## 用户可以输入的TensorRT 搜索路径
#set(TensorRT_ROOT
#	""
#	CACHE
#	PATH
#	${TRT_DIR})

## 设置TensorRT 搜索路径
#set(TensorRT_SEARCH_PATH
#  #/usr/include/x86_64-linux-gnu
#  #/usr/src/tensorrt
#  #/usr/lib/x86_64-linux-gnu
#  ${TensorRT_ROOT}
#)

## 设置需要搜索的TensorRT 依赖库
#set(TensorRT_ALL_LIBS
#  nvinfer
#  nvinfer_plugin
#  nvparsers
#  nvonnxparser
#)

## 提前设置后面需要用的变量
#set(TensorRT_LIBS_LIST)
#set(TensorRT_LIBRARIES)

## 搜索头文件的路径
#find_path(
#  TensorRT_INCLUDE_DIR
#  NAMES NvInfer.h
#  PATHS ${TensorRT_SEARCH_PATH}
#)
#set(TensorRT_INCLUDE_DIR ${TRT_INCLUDE_DIR})

## 利用头文件路径下的version文件来设置TensorRT的版本信息
#if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
#  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
#  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
#  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")
#
#  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
#  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
#  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
#  set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
#endif()
#message("TensorRT version: ${TensorRT_VERSION_STRING}")

## 搜索sample code的路径
#find_path(
#  TensorRT_SAMPLE_DIR
#  NAMES trtexec/trtexec.cpp
#  PATHS ${TensorRT_SEARCH_PATH}
#  PATH_SUFFIXES samples
#)

## 依次搜索TensorRT依赖库
#foreach(lib ${TensorRT_ALL_LIBS} )
#  find_library(
#    TensorRT_${lib}_LIBRARY
#    NAMES ${lib}
#    PATHS ${TensorRT_SEARCH_PATH}
#  )
#  ## 存储TensorRT的依赖库变量
#  set(TensorRT_LIBS_VARS TensorRT_${lib}_LIBRARY ${TensorRT_LIBS_LIST})
#  ## 也是TensorRT的依赖库，存成list，方便后面用foreach
#  list(APPEND TensorRT_LIBS_LIST TensorRT_${lib}_LIBRARY)
#endforeach()

## 调用cmake内置功能，设置基础变量如xxx_FOUND
find_package_handle_standard_args(TensorRT 
REQUIRED_VARS TRT_INCLUDE_DIR TRT_LIBRARY_DIR  TRT_LIBRARY TRT_SAMPLE_DIR
)

#if(TensorRT_FOUND)
#  ## 设置Tensor_LIBRARIES变量
#  foreach(lib ${TensorRT_LIBS_LIST} )
#    list(APPEND TensorRT_LIBRARIES ${${lib}})
#  endforeach()
#  message("Found TensorRT: ${TensorRT_INCLUDE_DIR} ${TensorRT_LIBRARIES} ${TensorRT_SAMPLE_DIR}")
#  message("TensorRT version: ${TensorRT_VERSION_STRING}")
#endif()


list(POP_BACK CMAKE_MESSAGE_INDENT)
