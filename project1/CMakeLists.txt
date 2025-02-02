# 1 project name
cmake_minimum_required(VERSION 3.15)
project(opencv_solver VERSION 0.1.0 LANGUAGES CXX)





# 2 构建第三方库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(fetch/spdlog)    # 构建了spdlog静态库
# 解决了spdlog.dll的问题;它是否会自动将找到的头文件路径放置到对应的头文件搜索路径中；lib文件路径放置到对应的libs搜索路径下

add_subdirectory(utils)  # 构建了logging接口库
#!? logging库在迁移到新项目时，需要根据新的日志记录库更换源文件中的相关代码；




# 3 发现并定义第三方库相关路径变量
set(THIRDPARTY_DIR     "D://Myself//MachineVision//third_party")
set(OpenCV_DIR         ${THIRDPARTY_DIR}/opencv455_DNN/install/x64/vc17/lib)
find_package(OpenCV    REQUIRED)   # 发现并定义opencv相关的路径变量





# 4 构建静态库
add_library(solver                STATIC  src/opencvSrc/src_101_120/A108.cpp)
target_include_directories(solver PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(solver      PRIVATE  logging ${OpenCV_LIBS} )





# 5 构建可执行文件
add_executable(opencv_solver             cli/main.cpp)
target_include_directories(opencv_solver PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_solver      PRIVATE  logging ${OpenCV_LIBS} solver )
#!? OpenCV_LIBS是不含路径的库文件名，链接器会自行在系统库目录等默认搜索目录中查找其所在位置(find_package(opencv)给到的路径)！





# 6 默认构建选项
set_target_properties(opencv_solver PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  DEBUG_POSTFIX "d"
)





# 7 调试
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_BINARY_DIR:   ${CMAKE_BINARY_DIR}")
message("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG:   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}")
message("OpenCV_INCLUDE_DIRS:   ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS:   ${OpenCV_LIBS}")
