[CMake生成Debug和Release目标程序时的一些配置_cmake debug-CSDN博客](https://blog.csdn.net/new9232/article/details/140567742)

设置Debug和Release目标生成时的bin输出目录

```cmake
set_target_properties(opencv_solver PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  DEBUG_POSTFIX "d"
)
```







# 使用说明

1. VisualStudio打开该项目，在项目-->CMake工作区设置中设置项目名称

   .vs/CMakeWorkspaceSettings.json中

   ```json
   {
     "enableCMake": true,
     "sourceDirectory": "project2"
   }
   ```







# 子项目解释

1. project1：opencv相关算子，生成静态库，main文件调用该静态库；
2. project2：opencv相关算子，生成动态库，main文件调用该动态库；
3. project3：onnxruntime+opencv+yolo相关算子，生成静态库，main文件调用该静态库；
4. project4：onnxruntime+opencv+yolo相关算子，生成动态库，main文件调用该动态库；
5. project9：历史遗留项目







# OnnxRuntime动态库

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









# 静态库的位置

```cmake
F:\Projects\Opencv-100-Questions\project3\out\build\x64-Debug\Debug\solver.lib  # 静态库文件
F:\Projects\Opencv-100-Questions\project3\include\onnx_deploy.h                 # 静态库头文件
```



# 动态库的位置

```cmake
F:\Projects\Opencv-100-Questions\project4\out\build\x64-Debug\Debug\solver.lib   # 动态库导入库文件
F:\Projects\Opencv-100-Questions\project4\out\build\x64-Debug\Debug\solver.dll   # 动态库文件
F:\Projects\Opencv-100-Questions\project4\out\build\x64-Debug\solver_export.h    # 导入库头文件
F:\Projects\Opencv-100-Questions\project4\include\onnx_deploy.h                  # 动态库头文件
```









# Project1项目设计

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