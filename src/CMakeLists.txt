
find_package(Qt5 COMPONENTS Core REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


include_directories(${OpenCV_INCLUDE_DIRS})
# 1 只有find_package(OpenCV REQUIRED)，OpenCV_INCLUDE_DIRS和OpenCV_LIBS才会被赋值。


# 查找所有源文件和头文件
file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")

#add_executable(My_Project "main.cpp" "A_1_10.h" "A_11_20.h" "A_21_30.h" "A_31_40.h" "A_41_50.h" "A_51_60.h" "A_61_70.h" "A_71_80.h" "A1.cpp" "A10.cpp" "A11.cpp" "A12.cpp" "A13.cpp" "A14.cpp" "A15.cpp" "A16.cpp" "A17.cpp" "A18.cpp" "A19.cpp" "A2.cpp" "A20.cpp" "A21.cpp" "A22.cpp" "A23.cpp" "A24.cpp" "A25.cpp" "A26.cpp" "A27.cpp" "A28.cpp" "A29.cpp" "A3.cpp" "A30.cpp" "A31.cpp" "A32.cpp" "A33.cpp" "A34.cpp" "A35.cpp" "A36.cpp" "A37.cpp" "A38.cpp" "A39.cpp" "A4.cpp" "A40.cpp" "A41.cpp" "A42.cpp" "A43.cpp" "A44.cpp" "A45.cpp" "A46.cpp" "A47.cpp" "A48.cpp" "A49.cpp" "A5.cpp" "A50.cpp" "A51.cpp" "A52.cpp" "A53.cpp" "A54.cpp" "A55.cpp" "A56.cpp" "A57.cpp" "A58.cpp" "A59.cpp" "A6.cpp" "A61.cpp" "A62.cpp" "A63.cpp" "A64.cpp" "A65.cpp" "A7.cpp" "A70.cpp" "A71.cpp" "A72.cpp" "A73.cpp" "A74.cpp" "A75.cpp" "A76.cpp" "A77.cpp" "A78.cpp" "A79.cpp" "A8.cpp" "A80.cpp" "A9.cpp")
add_executable(My_Project ${SOURCES} ${HEADERS})
# 1 执行文件仅使用一个cpp

target_link_libraries(My_Project ${OpenCV_LIBS})  
# 1 该行必须在add_executable后面，即My_Project必须生成了exe文件之后
# 2 如果没有这句话，opencv的命令会在生成阶段，报“LNK2019 : 无法解析的外部符号”的错误