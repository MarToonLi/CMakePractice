# 1 project name
cmake_minimum_required(VERSION 3.15)
project(opencv_solver VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)




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
find_package(OpenMP    REQUIRED)   # 发现并定义opencv相关的路径变量





# 4 构建静态库(多个！)
#add_library(solver110                STATIC  src/opencvSrc/src_101_120/A110.cpp)
#target_include_directories(solver110 PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(solver110      PRIVATE  logging ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


#add_library(solver111                STATIC  src/opencvSrc/src_101_120/A111.cpp)
#target_include_directories(solver111 PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(solver111      PRIVATE  logging ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


#add_library(solver112                STATIC  src/opencvSrc/src_101_120/A112.cpp)
#target_include_directories(solver112 PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} 
#    ${THIRDPARTY_DIR}/eigen3/install
#    ${THIRDPARTY_DIR}/eigen3/install/include
#    ${THIRDPARTY_DIR}/eigen3/install/include/eigen3
#    ${THIRDPARTY_DIR}/eigen3/install/include/eigen3/Eigen/src)
#target_link_libraries(solver112      PRIVATE  logging ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
# 这里需要注意的是，eigen在solver112的源文件中被使用，就需要在solver112下include其库文件；在opencv_solver中包含是没有用的；

add_library(solver113                STATIC  src/opencvSrc/src_101_120/A113.cpp)
target_include_directories(solver113 PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} 
    ${THIRDPARTY_DIR}/eigen3/install
    ${THIRDPARTY_DIR}/eigen3/install/include
    ${THIRDPARTY_DIR}/eigen3/install/include/eigen3
    ${THIRDPARTY_DIR}/eigen3/install/include/eigen3/Eigen/src)
target_link_libraries(solver113      PRIVATE  logging ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


add_library(solver114                STATIC  src/opencvSrc/src_101_120/A114.cpp)
target_include_directories(solver114 PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} 
    ${THIRDPARTY_DIR}/eigen3/install
    ${THIRDPARTY_DIR}/eigen3/install/include
    ${THIRDPARTY_DIR}/eigen3/install/include/eigen3
    ${THIRDPARTY_DIR}/eigen3/install/include/eigen3/Eigen/src)
target_link_libraries(solver114      PRIVATE  logging ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


# 5 构建可执行文件
add_executable(opencv_solver             cli/main.cpp)
target_include_directories(opencv_solver PUBLIC  include ${CMAKE_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_solver      
PRIVATE  
logging
${OpenCV_LIBS}  
OpenMP::OpenMP_CXX 
#solver110 
#solver111
#solver112
solver113
solver114
)
#!? OpenCV_LIBS是不含路径的库文件名，链接器会自行在系统库目录等默认搜索目录中查找其所在位置(find_package(opencv)给到的路径)！





# 6 默认构建选项
set_target_properties(opencv_solver PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  DEBUG_POSTFIX "d"
)



if(OpenCV_FOUND)
    # 查找所有可能的DLL路径
    get_filename_component(OpenCV_BIN_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)
    
    # OpenCV_VERSION 为4.5.5, 下述正则表达式可以将字符串中的点去除！
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\1\\2\\3" OpenCV_VERSION_3DIGIT ${OpenCV_VERSION})

    # 将opencv_worldxxx.dll文件拷贝到工作目录下
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${OpenCV_BIN_DIR}/opencv_world${OpenCV_VERSION_3DIGIT}$<$<CONFIG:Debug>:d>.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying OpenCV DLLs"
    )

endif()




# 7 调试
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_BINARY_DIR:   ${CMAKE_BINARY_DIR}")
message("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG:   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}")
message("OpenCV_INCLUDE_DIRS:   ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS:   ${OpenCV_LIBS}")
