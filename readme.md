# CMake Practice 使用说明

本项目旨在构建基于opencv传统算子和ONNX模型部署的静态库和动态库的CMake项目。

注：本项目虽然名为Opencv-100-Questions，但项目结构为多Cmake项目，前两个项目是将Opencv-100-Questions的源文件整理成了Cmake项目，方便新算子的探索和编写。



## VisualStudio配置待运行项目

1. VisualStudio打开该项目，在项目-->CMake工作区设置中设置项目名称

   .vs/CMakeWorkspaceSettings.json中

   ```json
   {
     "enableCMake": true,
     "sourceDirectory": "project2"
   }
   ```



## 各项目解释

1. project1：opencv相关算子，生成静态库，main文件调用该静态库；
2. project2：opencv相关算子，生成动态库，main文件调用该动态库；
3. project3：onnxruntime+opencv+yolo相关算子，生成静态库，main文件调用该静态库；
4. project4：onnxruntime+opencv+yolo相关算子，生成动态库，main文件调用该动态库；
5. project9：历史遗留项目





## 第三方库和必要dll文件

通过网盘分享的文件：Opencv-100-Questions.rar
链接: https://pan.baidu.com/s/1YlP07xg0EfWiWI2pT3exlw?pwd=gbnj 提取码: gbnj 
--来自百度网盘超级会员v7的分享



- third_party放置到任意文件夹，对应地需要修改cmakelist中对应变量的路径
- resources与各项目同目录；

**注：因为项目编写过程存在变动，有些资源的路径存在变动，因此项目运行可能存在问题；**





# 构建目标调用静态库和动态库一般流程

1. 声明静态库和动态库名称，及对应源文件
2. 添加头文件搜索目录
3. 链接动态库/动态库/导入库文件所在目录

示例代码如下：

```cmake
# ============================================ 4 构建静态库
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


# ================================================== 4 构建动态库
add_library(solver  SHARED  
${PROJECT_SOURCE_DIR}/src/onnxSrc/onnx_deploy.cpp  
${PROJECT_SOURCE_DIR}/src/onnxSrc/yolov5.cpp)     # 4.1 声明动态库名称和源文件
include(GenerateExportHeader)
generate_export_header(solver)                                                    # 4.2 构建头文件且头文件被源文件include(否则无法生成solver.lib)
target_compile_definitions(solver PRIVATE  solver_EXPORTS)                        # 4.2 (optional)为solver设置编译器宏定义solver_EXPORTS


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
```







# 静态库/动态库相关文件存储位置

```cmake
# 静态库--------------------------------------------
F:\Projects\Opencv-100-Questions\project3\out\build\x64-Debug\Debug\solver.lib  # 静态库文件
F:\Projects\Opencv-100-Questions\project3\include\onnx_deploy.h                 # 静态库头文件

# 动态库--------------------------------------------
F:\Projects\Opencv-100-Questions\project4\out\build\x64-Debug\Debug\solver.lib   # 动态库导入库文件
F:\Projects\Opencv-100-Questions\project4\out\build\x64-Debug\Debug\solver.dll   # 动态库文件
F:\Projects\Opencv-100-Questions\project4\out\build\x64-Debug\solver_export.h    # 导入库头文件
F:\Projects\Opencv-100-Questions\project4\include\onnx_deploy.h                  # 动态库头文件
```







# ProjectX项目设计流程

## 1 前期设计

### 1.1 模块设计

项目目标：快速地对opencv相关问题进行单元研发和测试。为此，项目需要定义两个构建目标：可执行文件目标opencv_solver和静态库目标solver。

### 1.2 项目目录结构

```cmake
project1
----Cmakelist.txt
----opencvSrc
----|----src_101_120
----|----|----A_101_120.h
----|----|----A108.cpp
----cmake/fetch
----|----spdlog.cmake
----images
----|----1.png ....
----cli
----|----main.cpp
----|----main.h
----utils
----|----logger.h
----|----Cmakelist.txt
```

### 1.3 接口设计

```c++
int main();
// ---------------
int A108_solver();
```





## 2 第三方库

### 2.1 安装库

### 2.2 库的查找模块



## 3 CMake目录程序

### 3.1 查找软件包

### 3.2 动态库目标

### 3.3 可执行文件目标



## 4 代码实现





# 额外小技巧

## 设置Debug和Release目标生成时的bin输出目录

[CMake生成Debug和Release目标程序时的一些配置_cmake debug-CSDN博客](https://blog.csdn.net/new9232/article/details/140567742)

```cmake
set_target_properties(opencv_solver PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  DEBUG_POSTFIX "d"
)
```



## OnnxRuntime动态库配置流程

1. CMakeLists.txt：`set(onnxruntime_ROOT      ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118)`

2. Findonnxruntime.cmake：

   ```cmake
   list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/fetch")
   find_package(onnxruntime  REQUIRED)
   
   list(APPEND CMAKE_MESSAGE_INDENT "[onnxtime]  ")
   
   set(onnxruntime_INCLUDE_DIR ${onnxruntime_ROOT}/include/onnxruntime/core/session              )
   set(onnxruntime_LIBRARY     ${onnxruntime_ROOT}/build/Windows/Release/Release/onnxruntime.lib )
   set(ENV{onnxruntime_INCLUDE_DIR} ${onnxruntime_ROOT}/include/onnxruntime/core/session              )
   set(ENV{onnxruntime_LIBRARY}     ${onnxruntime_ROOT}/build/Windows/Release/Release/onnxruntime.lib )
   set(onnxruntime_VERSION "1.17.0")
   set(onnxruntime_FOUND 1)
   ```

   - **.cmake文件中的cmake变量如果希望被目录文件中应用，需要使用ENV，而不能使用PARENT_SCOPE;**

3. CMakeLists.txt：**add_library的源文件中不需要加入头文件；**

4. CMakeLists.txt：target_include_directories中加入$ENV{onnxruntime_INCLUDE_DIR}；

5. CMakeLists.txt：target_link_libraries中加入 $ENV{onnxruntime_LIBRARY}；

6. **（🎯optional）S4和S5可以切换成target_link_libraries中加入onnxruntime::onnxruntime；**

7. 可执行文件所在目录中需要加入onnxruntime.dll和opencv_world455d.dll和solver.dll的文件；

8. **动态库方式的项目，问题在于，每一次动态库源文件的修改，都需要将对应的动态库文件粘贴复制到可执行文件所在目录中；**





# 参考

- 《CMake构建实战：项目开发卷》

- [全网最细的CMake教程！(强烈建议收藏) - 知乎](https://zhuanlan.zhihu.com/p/534439206)

