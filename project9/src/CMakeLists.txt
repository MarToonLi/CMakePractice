# 1 找到包安装目录
find_package(Qt5 COMPONENTS Core REQUIRED)
find_package(OpenCV REQUIRED)                # find_package会自动找到OpenCV相关路径的变量

#add_subdirectory(onnxSrc)
#add_subdirectory(onnxSrcDll)
add_subdirectory(utils)


# 2 包含目录增加内容
include_directories(
PUBLIC
    ./
    ${spdlog_INCLUDE_DIR}
    #"onnxSrcDll/typec_ocr_prepostprocess.h"
    "onnxSrcDll/typec_ocr.h"

    # opencv相关头文件
    "opencvSrc/src_101_120/A_101_120.h"

    # utils相关头文件
PRIVATE
    ./
    ${OpenCV_INCLUDE_DIRS}                                                        # 3.1 add opencv xxx.h
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include                            # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include/onnxruntime/core/session   
    ${THIRDPARTY_DIR}/cuda118/include                                             # 3.3 add cuda
)


# 3 构建目标
add_executable(
    My_Project 
    main.cpp
    "main.h"
   
    # opencv相关源文件
    "opencvSrc/src_101_120/A108.cpp"

    # 动态库和静态库的头文件不该出现在这里
)



# 3 添加附加依赖项目录
target_link_directories( My_Project
PUBLIC
)


# 4 添加库
target_link_libraries( My_Project 
PUBLIC
    # 共用内容
    logging
    ${OpenCV_LIBS}                                                                            # 3.1 add opencv xxx.lib
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/build/Windows/Release/Release/onnxruntime.lib  # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/cuda118/lib//x64//nvml.lib                                              # 3.3 add cuda
    
    # 源文件
    #onnxDeploy
    #onnxDeployDll

)  
#!? 1 该行必须在add_executable后面，即My_Project必须生成了exe文件之后
#!? 2 如果没有这句话，opencv的命令会在生成阶段，报“LNK2019 : 无法解析的外部符号”的错误
