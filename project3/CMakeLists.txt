# 1 project name
cmake_minimum_required(VERSION 3.15)
project(onnx_solver_shared VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)





# 2 发现并定义第三方库相关路径变量
set(THIRDPARTY_DIR        D:/Myself/MachineVision/third_party)                     #!? 不可以是//
set(OpenCV_DIR            ${THIRDPARTY_DIR}/opencv455_DNN/install/x64/vc17/lib)
set(onnxruntime_ROOT      ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118)
set(TRT_DIR               ${THIRDPARTY_DIR}/TensorRT-8.5.2.2-win)  
set(cuda_ROOT             ${THIRDPARTY_DIR}/cuda118)
#set(cuda_ROOT             "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")




# 3 构建第三方库
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/fetch") # F:/Projects/Opencv-100-Questions/project3 
include(spdlog)    # 构建了spdlog静态库
# 解决了spdlog.dll的问题;它是否会自动将找到的头文件路径放置到对应的头文件搜索路径中；lib文件路径放置到对应的libs搜索路径下
add_subdirectory(utils)  # 构建了logging接口库
#!? logging库在迁移到新项目时，需要根据新的日志记录库更换源文件中的相关代码；


find_package(OpenCV       REQUIRED)   # 发现并定义opencv相关的路径变量
find_package(onnxruntime  REQUIRED)   # 发现并定义onnxruntime相关的路径变量 ==> #!? 对应的cmake文件是"Findonnxruntime.cmake"
find_package(cuda         REQUIRED)   # 名字只能时cuda，不能是CUDA; 同时需要设定cuda_ROOT
#find_package(TRT_DIR      REQUIRED)  #! windows/linux版本下不存在cmake文件

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")





# 4 构建模型的plugin
add_library(mypluginssss1  SHARED 
${PROJECT_SOURCE_DIR}/src/plugin/yololayer.cu
)

include(GenerateExportHeader)
generate_export_header(mypluginssss1)                                                   
target_compile_definitions(mypluginssss1 PRIVATE  mypluginssss1_EXPORTS)  

target_include_directories(mypluginssss1  
PUBLIC
	include
	${cuda_INCLUDE_DIR}
	${TRT_DIR}/include
	${cuda_ROOT}/include
	${PROJECT_SOURCE_DIR}/src/plugin
) 
target_link_directories(mypluginssss1 
PUBLIC  
	${TRT_DIR}/lib
	${cuda_ROOT}/lib
	${cuda_ROOT}/lib/x64
)

target_link_libraries(mypluginssss1 nvinfer cudart)

set_property(TARGET mypluginssss1 PROPERTY CUDA_ARCHITECTURES 50 61 72 75)

FIND_PATH(mypluginssss1_INCLUDE_DIR yololayer.h      ${PROJECT_SOURCE_DIR}/src/plugin)
FIND_LIBRARY(mypluginssss1_LIBRARY mypluginssss1.lib ${CMAKE_BINARY_DIR}/Debug)




# 4 构建静态库
add_library(solver  STATIC  
	${PROJECT_SOURCE_DIR}/src/onnxSrc/onnx_deploy.cpp  
	${PROJECT_SOURCE_DIR}/src/onnxSrc/yolov5.cpp 
)

target_include_directories(solver  
PUBLIC  
	include 
	${CMAKE_BINARY_DIR}     # F:/Projects/CMakePractice/project3/out/build/x64-Debug
	${OpenCV_INCLUDE_DIRS} 
	${cuda_INCLUDE_DIR}
	${TRT_DIR}/include
	${cuda_ROOT}/include
	${PROJECT_SOURCE_DIR}/bin/debug
	${PROJECT_SOURCE_DIR}/src/plugin
	${mypluginssss1_INCLUDE_DIR}

)

target_link_directories(solver 
PUBLIC  
	${TRT_DIR}/lib
	${cuda_ROOT}/lib
	${cuda_ROOT}/lib/x64
	${CMAKE_BINARY_DIR}
	${CMAKE_BINARY_DIR}/Debug
)

target_link_libraries(solver  
PUBLIC  
	slogging 
	scommon
	onnxruntime::onnxruntime
	cuda
	nvinfer
	cudart
	nvinfer_plugin
	nvonnxparser
	nvparsers
	mypluginssss1
	${OpenCV_LIBS} 
	${mypluginssss1_LIBRARY}
) 



# 5 构建可执行文件${CMAKE_BINARY_DIR}/solver libs
add_executable(onnx_solver_shared                cli/main.cpp )
target_include_directories(onnx_solver_shared 
PUBLIC  
	include 
	${CMAKE_BINARY_DIR} 
	${OpenCV_INCLUDE_DIRS} 
	${CMAKE_SOURCE_DIR}/include

	${cuda_INCLUDE_DIR}
	${TRT_DIR}/samples/common
	${TRT_DIR}/include
	${cuda_ROOT}/include

	${PROJECT_SOURCE_DIR}/bin/debug
	${PROJECT_SOURCE_DIR}/src/plugin
	${mypluginssss1_INCLUDE_DIR}
)
target_link_directories(onnx_solver_shared 
PUBLIC  
	${TRT_DIR}/lib
	${cuda_ROOT}/lib
	${cuda_ROOT}/lib/x64
	${PROJECT_SOURCE_DIR}/bin/debug
	${CMAKE_BINARY_DIR}
	${CMAKE_BINARY_DIR}/Debug
)
target_link_libraries(onnx_solver_shared      
PUBLIC  
	mypluginssss1
	${OpenCV_LIBS} 
	${mypluginssss1_LIBRARY}
	solver
)



# 6 默认构建选项
#set_target_properties(mypluginssss1  PROPERTIES                      # 设置动态库输出路径
#    LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/bin/$<CONFIG>
#    RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/bin/$<CONFIG>
#)

set_target_properties(solver PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug   # 静态库文件输出目录
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  
  LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${PROJECT_SOURCE_DIR}/bin/debug  # 动态库文件输出目录
  LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/bin/release
)

set_target_properties(onnx_solver_shared PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug    # 可执行文件输出目录
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  DEBUG_POSTFIX "d"
)



# 编译附加项
add_compile_options(
    $<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<OR:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>>:/utf-8>
)





# 复制dll文件
#if(OpenCV_FOUND)
#    get_filename_component(OpenCV_BIN_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)  # 查找所有可能的DLL路径
#    string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\1\\2\\3" OpenCV_VERSION_3DIGIT ${OpenCV_VERSION})  # OpenCV_VERSION 为4.5.5, 正则将字符串中的点去除！
#    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD   # 将opencv_worldxxx.dll文件拷贝到工作目录下
#        COMMAND ${CMAKE_COMMAND} -E copy
#            "${OpenCV_BIN_DIR}/opencv_world${OpenCV_VERSION_3DIGIT}$<$<CONFIG:Debug>:d>.dll"
#            $<TARGET_FILE_DIR:${PROJECT_NAME}>
#        COMMENT "Copying OpenCV DLLs"
#    )
#endif()





# 7 调试
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_BINARY_DIR:   ${CMAKE_BINARY_DIR}")
message("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG:   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}")
message("OpenCV_INCLUDE_DIRS:   ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS:           ${OpenCV_LIBS}")
message("PROJECT_BINARY_DIR:    ${PROJECT_BINARY_DIR}")
message("onnxruntime_LIBRARY:       ${onnxruntime_LIBRARY}")
message("ENV.onnxruntime_LIBRARY:   $ENV{onnxruntime_LIBRARY}")
message("CMAKE_CUDA_COMPILER:   ${CMAKE_CUDA_COMPILER}")
message("cuda_INCLUDE_DIR:   ${cuda_INCLUDE_DIR}")
message("cuda_LIBRARIES:   ${cuda_LIBRARIES}")
message("CUDA_LIBRARIES:   ${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:   ${CUDA_INCLUDE_DIRS}")
