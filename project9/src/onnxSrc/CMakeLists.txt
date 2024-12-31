# 1 找到包安装目录
find_package(Qt5 COMPONENTS Core REQUIRED)
find_package(OpenCV REQUIRED)                # find_package会自动找到OpenCV相关路径的变量

aux_source_directory(. DIR_LIB_SRCS)

add_library(onnxDeploy ${DIR_LIB_SRCS})

# 2 包含目录增加内容
target_include_directories(
    onnxDeploy
PUBLIC
    ./
    ${spdlog_INCLUDE_DIR}
PRIVATE
    ./
    ${OpenCV_INCLUDE_DIRS}                                                        # 3.1 add opencv xxx.h
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include                            # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include/onnxruntime/core/session   
    ${THIRDPARTY_DIR}/cuda118/include                                             # 3.3 add cuda
)




# 3 库目录增加内容
target_link_libraries(
    onnxDeploy 
PUBLIC
    ${OpenCV_LIBS}                                                                            # 3.1 add opencv xxx.lib
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/build/Windows/Release/Release/onnxruntime.lib  # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/cuda118/lib//x64//nvml.lib                                              # 3.3 add cuda
)  


