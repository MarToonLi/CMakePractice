# 1 project name
cmake_minimum_required(VERSION 3.15)
project(onnx_solver_shared VERSION 0.1.0 LANGUAGES CXX)




# 2 发现并定义第三方库相关路径变量
set(THIRDPARTY_DIR        D:/Myself/MachineVision/third_party)                     #!? 不可以是//
set(OpenCV_DIR            ${THIRDPARTY_DIR}/opencv455_DNN/install/x64/vc17/lib)
set(onnxruntime_ROOT      ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118)
set(cuda_ROOT             ${THIRDPARTY_DIR}/cuda118)





# 3 构建第三方库
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/fetch") # F:/Projects/Opencv-100-Questions/project3 
include(spdlog)    # 构建了spdlog静态库
# 解决了spdlog.dll的问题;它是否会自动将找到的头文件路径放置到对应的头文件搜索路径中；lib文件路径放置到对应的libs搜索路径下
add_subdirectory(utils)  # 构建了logging接口库
#!? logging库在迁移到新项目时，需要根据新的日志记录库更换源文件中的相关代码；

find_package(OpenCV       REQUIRED)   # 发现并定义opencv相关的路径变量
find_package(onnxruntime  REQUIRED)   # 发现并定义onnxruntime相关的路径变量 ==> #!? 对应的cmake文件是"Findonnxruntime.cmake"
find_package(cuda         REQUIRED)   






# 4 构建静态库
add_library(solver  STATIC  
${PROJECT_SOURCE_DIR}/src/onnxSrc/onnx_deploy.cpp  
${PROJECT_SOURCE_DIR}/src/onnxSrc/yolov5.cpp)     # 4.1 声明静态库名称和源文件

target_include_directories(solver  
PUBLIC  
	include 
	${CMAKE_BINARY_DIR} 
	${OpenCV_INCLUDE_DIRS} 
	#$ENV{onnxruntime_INCLUDE_DIR} 
	${cuda_INCLUDE_DIR}
)   # 4.3 添加头文件目录
target_link_libraries(solver      
PUBLIC  
	logging 
	${OpenCV_LIBS} 
	#$ENV{onnxruntime_LIBRARY}
	onnxruntime::onnxruntime
	cuda
)   # 4.4 链接导入库文件和接入库

set_target_properties(solver PROPERTIES
  #!? 静态库文件输出目录
  ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release

  #!? 动态库文件输出目录
  LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${PROJECT_SOURCE_DIR}/bin/debug
  LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/bin/release
)




# 5 构建可执行文件${CMAKE_BINARY_DIR}/solver.libs
add_executable(onnx_solver_shared                cli/main.cpp)
target_include_directories(onnx_solver_shared 
PUBLIC  
	include 
	${CMAKE_BINARY_DIR} 
	${OpenCV_INCLUDE_DIRS} 
	${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(onnx_solver_shared      
PRIVATE  
	logging 
	${OpenCV_LIBS}  
	solver
)
#!? OpenCV_LIBS是不含路径的库文件名，链接器会自行在系统库目录等默认搜索目录中查找其所在位置(find_package(opencv)给到的路径)！





# 6 默认构建选项
set_target_properties(onnx_solver_shared PROPERTIES
  # 可执行文件输出目录
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug   
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release

  DEBUG_POSTFIX "d"
)






# 7 调试
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_BINARY_DIR:   ${CMAKE_BINARY_DIR}")
message("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG:   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}")
message("OpenCV_INCLUDE_DIRS:   ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS:           ${OpenCV_LIBS}")
message("PROJECT_BINARY_DIR:    ${PROJECT_BINARY_DIR}")
message("onnxruntime_LIBRARY:       ${onnxruntime_LIBRARY}")
message("ENV.onnxruntime_LIBRARY:   $ENV{onnxruntime_LIBRARY}")
