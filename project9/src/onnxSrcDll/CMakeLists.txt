# 1 找到包安装目录
find_package(Qt5 COMPONENTS Core REQUIRED)
find_package(OpenCV REQUIRED)                # find_package会自动找到OpenCV相关路径的变量

aux_source_directory(. DLL_DIR_LIB_SRCS)

add_library(onnxDeployDll ${DLL_DIR_LIB_SRCS})

# 2 包含目录增加内容
target_include_directories(
    onnxDeployDll
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


message(status "==> ${CMAKE_CURRENT_LIST_DIR}")   #  F:/Projects/CMakeProjects/src/ch011

# 3 库目录增加内容
target_link_libraries(
    onnxDeployDll 

PUBLIC
    ${OpenCV_LIBS}                                                                            # 3.1 add opencv xxx.lib
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/build/Windows/Release/Release/onnxruntime.lib  # 3.2 add onnxtime
    ${THIRDPARTY_DIR}/cuda118/lib//x64//nvml.lib                                              # 3.3 add cuda

    debug ${CMAKE_CURRENT_LIST_DIR}/typec_ocr_prepostprocess_D.lib
    optimized ${CMAKE_CURRENT_LIST_DIR}/typec_ocr_prepostprocess_R.lib
    ${CMAKE_CURRENT_LIST_DIR}/typec_ocr_run.lib
)  


