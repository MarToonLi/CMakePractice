# 1 找到包安装目录
find_package(Qt5 COMPONENTS Core REQUIRED)
find_package(OpenCV REQUIRED)


# 2 包含目录增加内容
include_directories(
PUBLIC
    ./
    ${spdlog_INCLUDE_DIR}
PRIVATE
    ./
    ${OpenCV_INCLUDE_DIRS}                                                        # 3.1 add opencv
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include                            # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include/onnxruntime/core/session   
    ${THIRDPARTY_DIR}/cuda118/include                                             # 3.3 add cuda
)


# 查找所有源文件和头文件
file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")

add_executable(
    My_Project 
    main.cpp
    "opencvSrc/src_101_120/A_101_120.h"
    "opencvSrc/src_101_120/A108.cpp"

    "onnxSrc/yolov5.cpp" 
    "onnxSrc/yolov5.hpp" 
    "onnxSrc/onnx_deploy.cpp"
    "onnxSrc/onnx_deploy.h"

    "utils/logger.h"
)


# 3 库目录增加内容
target_link_libraries(
    My_Project 
    ${OpenCV_LIBS}                                                                            # 3.1 add opencv
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/build/Windows/Release/Release/onnxruntime.lib  # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/cuda118/lib//x64//nvml.lib                                              # 3.3 add cuda
)  
#!? 1 该行必须在add_executable后面，即My_Project必须生成了exe文件之后
#!? 2 如果没有这句话，opencv的命令会在生成阶段，报“LNK2019 : 无法解析的外部符号”的错误


# 4 debug test for study
message(STATUS "CMake Variables:\n")
get_cmake_property(VAR_NAMES VARIABLES)  # 获取所有已定义的 CMake 变量
set(SEARCH_STRING "spdlog")              # 定义你想查找的字符或子字符串
foreach(VAR_NAME ${VAR_NAMES})           # 遍历所有变量名，查找是否包含指定的子字符串
    string(FIND ${VAR_NAME} ${SEARCH_STRING} POSITION)
    if(POSITION GREATER -1)  # 如果找到了该字符或子字符串
        message(STATUS "==> Found variable: ${VAR_NAME}")
    endif()
endforeach()


message(STATUS "lib: ${OpenCV_LIBRARY}")        # 空
message(STATUS "inc: ${OpenCV_INCLUDE_DIR}")    # 空
message(STATUS "inc: ${OpenCV_INCLUDE_DIRS}")   # 有
message(STATUS "inc: ${OpenCV_LIBRARIES}")      # 有
message(STATUS "inc: ${OpenCV_LIBRARY_DIRS}")   # 无
message(STATUS "inc: ${OpenCV_ROOT_DIR}")       # 无
message(STATUS "inc: ${OpenCV_FOUND}")          # 1



message(STATUS "THIRDPARTY_DIR: ${THIRDPARTY_DIR}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "spdlog_INCLUDE_DIR: ${spdlog_INCLUDE_DIR}")

MESSAGE(STATUS "Check opencv version: ${OpenCV_VERSION}")  # 必要核实