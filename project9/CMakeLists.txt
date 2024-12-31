# 1 project name
cmake_minimum_required(VERSION 3.15)
project(My_Project VERSION 0.1.0 LANGUAGES CXX)


# 2 指定各包中含有xxx.cmake文件的路径
if(DEFINED THIRDPARTY_DIR)
    #set(OpenCV_DIR ${THIRDPARTY_DIR}/opencv455_DNN/install/x64/vc17/lib)
    set(OpenCV_DIR F://CmakeProjects/opencv-4.5.5/build_x64_2/install/x64/vc17/lib)
    set(spdlog_INCLUDE_DIR ${THIRDPARTY_DIR}/spdlog/include)
endif()


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(fetch/spdlog) 
# 解决了spdlog.dll的问题;它是否会自动将找到的头文件路径放置到对应的头文件搜索路径中；lib文件路径放置到对应的libs搜索路径下


# 3 编译器相关的选项
if(MSVC)
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING )
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${PROJECT_SOURCE_DIR}/bin/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release)

add_compile_options(
    $<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<OR:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>>:/utf-8>
)


# 4 包含子目录
add_subdirectory(src)


message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "THIRDPARTY_DIR src dir : ${THIRDPARTY_DIR}")
message(STATUS "WIKKYDLL_DIR src dir : ${WIKKYDLL_DIR}")


