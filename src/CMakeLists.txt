
find_package(Qt5 COMPONENTS Core REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


include_directories(${OpenCV_INCLUDE_DIRS})
# 1 只有find_package(OpenCV REQUIRED)，OpenCV_INCLUDE_DIRS和OpenCV_LIBS才会被赋值。


# 查找所有源文件和头文件
file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")

add_executable(My_Project "main.cpp" "A_71_80.h" "A79.cpp" )
#add_executable(My_Project ${SOURCES} ${HEADERS})
#add_executable(My_Project ${HEADERS})

# 1 执行文件仅使用一个cpp

target_link_libraries(My_Project ${OpenCV_LIBS})  
# 1 该行必须在add_executable后面，即My_Project必须生成了exe文件之后
# 2 如果没有这句话，opencv的命令会在生成阶段，报“LNK2019 : 无法解析的外部符号”的错误