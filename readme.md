# 一 CMake Practice 项目使用说明

本项目旨在构建基于opencv传统算子和ONNX模型部署的静态库和动态库的CMake项目。

注：本项目虽然名为Opencv-100-Questions，但项目结构为多Cmake项目，前两个项目是将Opencv-100-Questions的源文件整理成了Cmake项目，方便新算子的探索和编写。





## 各项目解释

1. project1：opencv相关算子，生成静态库，main文件调用该静态库；
2. project2：opencv相关算子，生成动态库，main文件调用该动态库；
3. project3：onnxruntime+opencv+yolo相关算子，生成静态库，main文件调用该静态库；
4. project4：onnxruntime+opencv+yolo相关算子，生成动态库，main文件调用该动态库；
5. project9：历史遗留项目



## VisualStudio配置与执行

### Step1: 下载和配置第三方库和必要dll文件

通过网盘分享的文件：Opencv-100-Questions.rar
链接: https://pan.baidu.com/s/1YlP07xg0EfWiWI2pT3exlw?pwd=gbnj 提取码: gbnj 
--来自百度网盘超级会员v7的分享

该压缩文件中包含以下两个文件夹third_party和resources。

- third_party放置到任意文件夹；
- resources与各项目同目录；



### Step2: 修改源文件中相关路径

- THIRDPARTY_DIR的路径：修改cmakelist中对应变量THIRDPARTY_DIR的路径；
- 模型路径：如果运行project3和project4这两类使用了onnx的项目，需要在onnx_deploy.cpp中修改onnxModelPath的路径，模型文件在resources中；—— 如果不配置，可能会报Ort:Exception的错误。

注：resources中包含了opencv和onnx的相关dll文件，最终项目执行的时候需要将相关dll文件放置到与可执行文件同目录下；



### Step3: VisualStudio配置选择待运行项目

1. VisualStudio打开该项目，在项目-->CMake工作区设置中设置项目名称

   （或者在.vs/CMakeWorkspaceSettings.json中修改，内容如下）

   ```json
   {
     "enableCMake": true,
     "sourceDirectory": "project3"  // 指定待运行的项目
   }
   ```

2. 对应项目配置

   以project3为例；

   - 选择x64_Debug或者x64_Release；

   - 选择启动项为onnx_solver_sharedd.exe；

   - 打开projects3/Cmakelist.txt文件，并在对应编辑界面执行ctrl+s，VS会自动对当前项目进行cmake编译。

   - 编译结束后，将相关dll文件放置到可执行文件所在目录中；

     以project3为例，需要将opencv_world和onnxruntimed.dll文件放置到对应目录中；

   - 点击启动项onnx_solver_sharedd.exe，项目启动运行；

   

   



# 二 Cmake项目构建相关知识整理



## 背景：c++代码编译生成流程、CMake项目编译生成流程和VS软件对CMake项目的操作流程分别是什么？以及三者流程步骤之间的匹配关系如何？

C++代码编译生成流程、CMake项目编译生成流程以及VS软件对CMake项目的操作流程各有其特点。理解这三者的流程及其匹配关系有助于更好地掌握C++开发环境的构建过程。下面我会详细介绍这三者的流程步骤以及它们之间的匹配关系。

### 1. C++代码编译生成流程
C++代码的编译和生成过程通常包括以下几个步骤：

1.1 `编写源代码`：源文件和头文件

1.2 `预处理阶段`（Preprocessing）：处理以#开头的预处理命令；

- 宏替换：所有的宏定义（`#define`）会被展开。
- 文件包含：通过`#include`指令包含的头文件会被插入到源文件中。
- 条件编译：`#ifdef`等条件编译指令根据是否满足特定条件来选择性地包含代码。

1.3 `编译阶段`（Compilation）：将预处理后的代码翻译成（汇编语言进而）机器语言（目标文件）

- 词法分析、语法分析：将预处理后的源代码转化为中间表示（如抽象语法树AST）。
- 生成目标代码：最终编译器会将源代码转化为目标文件（.o 或 .obj），这些目标文件包含了机器指令，但还不能直接执行。

1.4 `链接阶段`（Linking）：将多个目标文件和库文件组合在一起，生成可执行文件（或库文件！📕）

### 2. CMake项目编译生成流程
CMake是一个跨平台的构建工具，它通过定义一个`CMakeLists.txt`文件来描述项目的构建规则。CMake项目的编译生成流程通常包括以下几个步骤：

2.1 `编写CMakeLists.txt`：核心配置文件，需要指定项目名称、源文件列表、头文件路径和链接库信息。

```cmake
project(MyProject)
add_executable(MyExecutable main.cpp other.cpp)
target_include_directories(MyExecutable PRIVATE include)
```

上述代码定义了一个名为MyProject的项目，生成一个可执行文件MyExecutable，包含main.cpp和other.cpp两个源文件，并且指定了头文件路径为include目录。

2.2 `运行CMake生成构建系统文件`：通过运行cmake命令，CMake会根据CMakeLists.txt文件生成平台相关的构建系统

（如unix系统下的Makefile、Windows系统下的Visual Studio解决方案.sln和.vcproj等）。

- linux系统中通常会人为创建build文件夹，然后在该文件夹中进行生成；
- windows系统中默认会在`生成根：${projectDir}\out\build\${name}文件夹下`生成；

2.3 `使用构建系统文件执行生成目标文件`：执行实际的编译、链接操作，生成库文件和可执行文件。

- linux下，如果是Makefile，可以运行make命令进行编译生成；

- Windows下，如果是Visual Studio 项目文件，可以使用 Visual Studio 打开并编译生成。

  `📕其中生成路径的指定和默认路径如下所示`：

  ```cmake
  # 下述代码中的两个属性指定了不同构建模式下（Debug 和 Release）的可执行文件（或库文件）的输出目录
  set_target_properties(名称 PROPERTIES
    # 名称是可执行文件，应用以下可执行文件输出目录
    # 默认是生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹下
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
    
    # 名称是静态库文件，应用以下可执行文件输出目录
    # 默认是生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹下
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  
    # 名称是动态库文件，应用以下可执行文件输出目录
    # 默认是生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹下
    LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${PROJECT_SOURCE_DIR}/bin/debug
    LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/bin/release
  )
  ```

### 3. Visual Studio (VS) 操作CMake项目的流程
Visual Studio可以通过内建的CMake支持直接打开和管理CMake项目。操作流程如下：

3.1 `打开CMake项目`

- 打开CMake项目：选择“文件”->“打开”->“CMake项目”，然后选择CMakeLists.txt文件，VS会自动识别并生成CMake的构建系统。
- 多CMake项目选择：文件夹下包含多个项目时，可以在“项目”-->“CMake工作区设置”中选择要执行的项目；

3.2 `配置CMake项目：VS菜单栏中的项目->xxxx的CMake设置`

**VS会自动调用CMake生成对应平台的构建文件**，用户也可以通过 项目->xxxx的CMake设置来修改配置选项。

- `📕配置选择项包含`：

  - 常规：
    - 配置名称：x64_Debug还是x64_Release；
    - 配置类型：Debug还是Release；
    - 工具集：msmv_x64_x64；
    - `生成根：${projectDir}\out\build\${name}`
  - 命令参数
    - CMake命令参数；
    - 生成命令参数；
  - CMake变量和缓存
  - 高级
    - CMake生成器，负责编写本机生成系统的输入文件：VisualStudio 17 2022 Win64；
    - `安装目录：${projectDir}\out\install\${name}`
    - CMake可执行文件；

- VS会处理CMake的配置并生成适合当前环境的构建系统文件（如.sln和.vcproj）。

  `Windows下，构建系统文件默认路径是：${projectDir}\out\build\${name}下，即3.2中提到的生成根`

3.3 `生成项目：VS菜单栏中的生成->全部生成`

- 编译与链接：在VS中点击“全部生成”按钮，它会使用生成的Visual Studio构建系统执行编译和链接操作，如果需要生成静态库文件或者动态库文件，则库文件默认会出现在 `生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹`。
- 调试与运行：点击“启动”按钮，VS会通过生成的可执行文件启动调试。此时，可执行文件默认会出现在上述提及的默认路径下。

### 总结
- **C++编译生成流程**：从源代码的编写到编译和链接，涉及预处理、编译、链接等步骤，最终生成可执行文件。
- **CMake项目编译生成流程**：通过编写`CMakeLists.txt`配置构建过程，然后通过CMake生成构建系统（如Makefile、VS工程等），最后使用这些构建系统来编译和链接项目。
- **Visual Studio操作CMake项目**：VS为CMake提供原生支持，通过界面操作直接管理CMake项目，包括配置、编译、调试等。

这三者之间的匹配关系在于：
- **C++编译生成流程**是所有构建过程的核心，而**CMake项目编译生成流程**是通过CMake工具化地自动化这些步骤，生成具体的构建系统（如Makefile或VS项目）。
- **Visual Studio**作为一个集成开发环境，它通过内建的CMake支持来简化和自动化这一过程，让开发者能够更直观地操作和管理CMake项目。







## 项目：ProjectX项目设计流程

1 前期设计

1.1 模块设计

项目目标：快速地对opencv相关问题进行单元研发和测试。为此，项目需要定义两个构建目标：可执行文件目标opencv_solver和静态库目标solver。

1.2 项目目录结构

```shell
project1
#!! 编写项目的Cmakelist.txt文件
----Cmakelist.txt         
#!! 源文件存储的位置
----opencvSrc
----|----src_101_120
----|----|----A_101_120.h
----|----|----A108.cpp
#!! 第三方库的cmake文件所在目录
# xxx.cmake文件用于从系统环境变量等路径中寻找到系统中安装的第三方库的安装路径，并发现对应的include、lib路径
# 1. 将include、lib路径配置成Cmake环境变量，供全局使用；2. 将相关的include、lib路径配置用于生成接口库，供全局使用；
----cmake/fetch
----|----spdlog.cmake
# resource资源
----images
----|----1.png ....
# 头文件目录
----cli
----|----main.cpp
----|----main.h
# utils：第三方库相关接口文件等任意内容
----utils
----|----logger.h
----|----Cmakelist.txt
```

1.3 接口设计

```c++
int main();
// ---------------
int A108_solver();
```

2 第三方库

2.1 安装库

2.2 库的查找模块

3 CMake目录程序

3.1 查找软件包

3.2 动态库目标

3.3 可执行文件目标

4 代码实现







## 库：构建目标调用静态库和动态库一般流程

1. 声明静态库和动态库名称，及对应源文件或者文件列表；
2. 添加头文件搜索目录；
3. 链接动态库/动态库/导入库文件的名称或者名称列表
4. (opt)将动态库文件拷贝到可执行文件所在目录；

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





## 库：静态库/动态库相关文件存储位置

以project3和project4项目为例

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



## 库：OnnxRuntime动态库配置流程

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



## 命令：target_link_libraries为什么有时不需要指定lib文件的路径

- 首先，target_link_libraries会自动从CMAKE_BINARY_DIR构建目录中或者CMAKE_SOURCE_DIR源代码目录下，搜索目标名称的lib文件。

- 其次，如果CMakelist.txt文件中包含了add_library命令，则Cmake会自动将该库的输出路径加入到CMake的库搜索路径中。

  以project3为例。输出路径是LIBRARY_OUTPUT_DIRECTORY指定的路径或者ARCHIVE_OUTPUT_DIRECTORY指定的路径



## 命令：find_package寻找库以及处理库的操作

`find_package(onnxruntime REQUIRED)` 是 CMake 中用来查找并配置 `onnxruntime` 库的一个命令。下面是它如何工作及执行后续操作的详细说明：

### 1. **如何搜索 `onnxruntime` 库的位置**
CMake 的 `find_package` 命令用来查找一个已安装的库或包。其查找过程通常分为以下几个步骤：

a. **检查默认路径**

- （主推）**CMake模块路径**：首先，CMake 会检查 `CMAKE_MODULE_PATH` 中指定的路径。可以人为添加搜索路径

  ```cmake
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/fetch") 
  ```

- （怀疑）**ONNXRUNTIME_DIR或类似环境变量**：若环境变量中定义了ONNXRUNTIME_DIR或类似的路径，CMake会优先在这些路径中查找。

- （次）**系统默认位置**：某些平台上，CMake会在标准的安装路径（如/usr/local, /opt, 或 Windows 上的特定安装目录）中查找。

b. **搜索 CMake 配置文件**

- **`Findonnxruntime.cmake`**：CMake 会查找名为 `Findonnxruntime.cmake` 的模块文件。这些文件通常位于系统的 CMake 模块目录下（`CMAKE_MODULE_PATH` 目录）或者在你项目的 `cmake` 子目录下。
  - 如果你没有自定义 `Findonnxruntime.cmake`，CMake 会尝试根据默认的搜索路径来查找 `onnxruntime` 的库位置。

c. **通过包的 CMake 配置文件**

- 如果 `onnxruntime` 已安装并包含自己的 CMake 配置文件（通常是 `onnxruntimeConfig.cmake` 或类似的文件），`find_package` 会直接查找这些文件。

### 2. **找到 `onnxruntime` 后要执行的操作**

当 `find_package(onnxruntime REQUIRED)` 找到 `onnxruntime` 后，它会**设置相关的 CMake 变量**

CMake 会设置一些变量，使你能够在项目中使用 `onnxruntime`：
- **`onnxruntime_INCLUDE_DIR`**：指向 `onnxruntime` 头文件的路径。你可以使用这些头文件来访问 `onnxruntime` 的 API。
- **`onnxruntime_LIBRARIES`** 或 **`onnxruntime::onnxruntime`**：这是库的路径和名称，用于链接 `onnxruntime` 库。
  - 这些库可以是 `.a`（静态库）或 `.so`（共享库），取决于你安装的 `onnxruntime` 类型。
- **`onnxruntime_VERSION`**：提供 `onnxruntime` 的版本信息。

**📕：在项目的根目录 `CMakeLists.txt` 文件中调用 `find_package`，并且该命令成功找到 `onnxruntime`，它会将相关变量设置为全局作用域（如果你没有明确指定作用域）。因此，根目录下的 `CMakeLists.txt` 文件可以直接访问这些变量**





## 命令：include(XXX)会搜索并执行xxx.cmake文件

在 CMake 中，`include` 语句的作用是将一个 CMake 脚本文件（通常是 `.cmake` 文件）包含到当前的 `CMakeLists.txt` 文件中，并执行该脚本文件中的内容。它允许你复用代码、模块化 CMake 配置，并在多个 CMake 项目或 CMakeLists 文件之间共享设置。

### `include` 语句的基本作用：

1. **执行包含的 CMake 脚本文件**
   - 当你在 `CMakeLists.txt` 中使用 `include(some_file.cmake)` 时，CMake 会查找并执行 `some_file.cmake` 中的所有 CMake 命令。这允许你在多个地方复用代码，不需要重复编写相同的配置或设置。
   
2. **扩展当前作用域**
   - `include` 是在当前的作用域内执行脚本文件的，因此包含的脚本文件对当前作用域的变量有影响。包含的脚本文件中的变量、函数和宏将会影响到包含它们的文件的后续内容。

3. **提供模块或函数**
   - 许多 CMake 库和模块（如 `FindXYZ.cmake` 或其他 CMake 模块）通常会使用 `include` 来引入特定的设置和配置函数。例如，CMake 的 `FindBoost.cmake` 脚本可以通过 `include` 引入，并在 CMake 配置过程中自动设置 Boost 库的路径和相关标志。

4. **查找并执行其他 CMake 文件**
   - `include` 会查找文件，如果没有找到该文件，CMake 会报错。如果文件路径相对不清晰，通常会使用 `CMAKE_MODULE_PATH` 来告诉 CMake 在哪些目录查找 CMake 模块和脚本文件。
   
### 举个例子：
```cmake
# 假设我们有一个 CMake 文件 external.cmake，其中包含一些设置
include(external.cmake)

# 在 external.cmake 中的内容将会影响到当前作用域
message("Value of variable from external.cmake: ${some_variable}")
```

### 更详细的行为：
- 如果你使用相对路径（如 `include(some_file.cmake)`），CMake 会尝试从当前 `CMakeLists.txt` 所在目录开始查找该文件。
- 如果你使用绝对路径（如 `include(/path/to/some_file.cmake)`），则直接使用该路径查找并执行文件。
- 如果文件名没有扩展名，CMake 默认会查找 `.cmake` 文件。

### 📕 总结：
- `include` 用于引入和执行其他 CMake 文件中的脚本、变量和函数。
- 它允许复用代码、扩展当前 CMake 配置文件，并在多个 `CMakeLists.txt` 文件之间共享设置。
- 通过 `include` 引入的文件中的设置会影响当前作用域，变量、宏、函数等将被导入到当前的 CMake 配置中。



## 解疑：下述两个代码的效果为什么一致

```cmake
target_include_directories(solver  
PUBLIC  
	include 
	${CMAKE_BINARY_DIR} 
	${OpenCV_INCLUDE_DIRS} 
	$ENV{onnxruntime_INCLUDE_DIR} 
	${cuda_INCLUDE_DIR}
)   # 4.3 添加头文件目录

target_link_libraries(solver      
PUBLIC  
	logging 
	${OpenCV_LIBS} 
	$ENV{onnxruntime_LIBRARY}
	#onnxruntime::onnxruntime
	cuda
)   # 4.4 链接导入库文件和接入库
# ==================================================
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

**因为onnxruntime::onnxruntime对头文件的添加方式是interface！**





## 命令：FetchContent_MakeAvailable

`FetchContent_Declare` 指定了一个外部库（`googletest`），并给出了它的 Git 仓库地址。`FetchContent_MakeAvailable` 会下载并构建这个库，之后可以将该外部库集成到目标（如 `my_test`）中使用它。

```cmake
# 引入 FetchContent 模块
include(FetchContent)

# 使用 FetchContent 下载并构建一个外部库，例如 googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.10.0
)

# 使用 FetchContent_MakeAvailable 来下载和构建 googletest
，(googletest)

# 现在可以使用 googletest 库进行链接和编译
add_executable(my_test test.cpp)
target_link_libraries(my_test googletest)
```



include和FetchContent_MakeAvailable的最大区别是，前者主要是复用一些变量、函数，而后者是构建一个完完整整的库。而该库是静态库还是动态库，则取决于googletest的cmakelist中如何定义库的类别。



## 📕 猜想：logger.h和spdlog的CMakeLists.txt预示

预示：如果A库通过target_link_libraries引用了B库目标，则A库在拥有B库的链接库文件列表的同时也会拥有B库的头文件列表。

—— 错误的

**Q：上述代码中的spdlog链接到logging 后，是否spdlog的头文件列表也会添加到logging 中**

```cmake
add_library(logging INTERFACE) 
target_sources(logging INTERFACE logger.h) 
arget_link_libraries(logging INTERFACE spdlog) 
target_include_directories(logging INTERFACE ./)
```



在你提供的 CMake 代码中，`logging` 是一个 **`INTERFACE`** 类型的库，这意味着它不会生成实际的目标文件（如 `.a` 或 `.so`），而是只传递接口信息给依赖它的目标。具体到你的问题，`spdlog` 的头文件列表是否会被添加到 `logging`，我们需要理解以下几点：

关键点分析：

1. **`INTERFACE` 库**：  
   `INTERFACE` 库本身不包含源代码和编译生成的文件，它只传递接口信息（如头文件、链接库等）给其他链接到它的目标。

2. **`target_link_libraries(logging INTERFACE spdlog)`**：  
   这一行的作用是将 `spdlog` 作为 `logging` 的依赖库，并以 `INTERFACE` 的方式链接它。这意味着，任何链接到 `logging` 库的目标（比如其他的库或可执行文件）会继承 `spdlog` 的链接库和头文件信息。

3. **`target_include_directories(logging INTERFACE ./)`**：  
   这行代码通过 `INTERFACE` 关键字将当前目录（`./`）的头文件目录添加到 `logging` 的接口中。所有链接到 `logging` 的目标都将会继承这个头文件路径。

4. **`spdlog` 的头文件**：  
   `target_link_libraries(logging INTERFACE spdlog)` 表示 `logging` 库将 `spdlog` 库作为接口依赖。`spdlog` 本身也是一个库（通常是一个头文件-only 的库），因此 `spdlog` 的头文件路径会传递给所有链接到 `logging` 库的目标，而不需要显式调用 `target_include_directories`。

结果分析：

- 当你链接 `spdlog` 到 `logging` 时，`logging` 的依赖目标会继承 `spdlog` 库的头文件路径（因为 `spdlog` 是一个 `INTERFACE` 库，通常它会通过 `INTERFACE` 或 `PUBLIC` 传递头文件路径）。
- 通过 `target_link_libraries(logging INTERFACE spdlog)`，`spdlog` 的头文件目录会自动添加到 `logging` 目标中。这意味着，任何依赖于 `logging` 的目标将自动获得 `spdlog` 的头文件路径。
- **总而言之**，是的，`spdlog` 的头文件列表会被添加到 `logging` 中，并传递给所有链接到 `logging` 的目标。

示例说明：

假设 `spdlog` 库在 CMake 中如下设置：
```cmake
add_library(spdlog INTERFACE)
target_include_directories(spdlog INTERFACE ${SPDLOG_INCLUDE_DIRS})
```

在这种情况下，`spdlog` 的头文件路径将作为接口属性传递。当你写：
```cmake
add_library(logging INTERFACE)
target_sources(logging INTERFACE logger.h)
target_link_libraries(logging INTERFACE spdlog)
target_include_directories(logging INTERFACE ./)
```
- `logging` 依赖于 `spdlog`，因此它将自动包含 `spdlog` 的头文件路径。
- 任何链接到 `logging` 的目标（例如其他库或可执行文件）都会继承 `spdlog` 和 `logging` 的头文件路径。

总结：

通过 `target_link_libraries(logging INTERFACE spdlog)`，`spdlog` 的头文件路径确实会添加到 `logging` 中，并且所有链接到 `logging` 的目标都能访问 `spdlog` 的头文件路径。



**悟了：之所以onnxruntime::onnxruntime的方式成立，是因为Findonnxruntime.cmake中onnxruntime::onnxruntime通过INTERFACE的方式链接到onnxruntime_INCLUDE_DIRS。**



### **📕 Q：下述代码的意思是什么？**

```cmake
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  target_include_directories(onnxruntime::onnxruntime INTERFACE ${onnxruntime_INCLUDE_DIRS})
  message("WIN32: ${WIN32}")
  if(WIN32)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES 
      IMPORTED_IMPLIB "${onnxruntime_LIBRARY}")
  else()
    set_target_properties(onnxruntime::onnxruntime PROPERTIES 
      IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
  endif()
```

这段 CMake 代码的目的是定义一个外部共享库目标 `onnxruntime::onnxruntime`，并根据平台设置该库的相关属性。代码还包括了一个条件判断部分，用于区分 Windows (`WIN32`) 平台和其他平台（如 Linux/macOS）来设置不同的库文件位置属性。

### 逐行解析：

1. **`add_library(onnxruntime::onnxruntime SHARED IMPORTED)`**
   - `add_library` 用来定义一个新的 CMake 库目标。
   - `onnxruntime::onnxruntime` 是库的名称，采用命名空间 `onnxruntime`，使其在 CMake 中具有更清晰的标识。
   - `SHARED` 表示这个库是一个 **共享库**（动态库），通常是 `.so`（Linux）、`.dll`（Windows）或 `.dylib`（macOS）文件。
   - `IMPORTED` 表示该库并不是由当前项目构建的，而是一个外部库，CMake 将通过其他途径来找到这个库并使用它。

   这行代码的作用是创建一个外部共享库目标 `onnxruntime::onnxruntime`，告诉 CMake 这个库的实际构建和链接由外部完成。

2. **`target_include_directories(onnxruntime::onnxruntime INTERFACE ${onnxruntime_INCLUDE_DIRS})`**
   - `target_include_directories` 设置目标的包含目录（头文件路径）。
   - `onnxruntime::onnxruntime` 是我们定义的外部库目标。
   - `INTERFACE` 表示这些头文件路径只会传递给依赖于 `onnxruntime::onnxruntime` 的其他目标，而不是目标本身。
   - `${onnxruntime_INCLUDE_DIRS}` 是一个 CMake 变量，包含了 ONNX Runtime 库的头文件路径，通常在其他地方的 CMake 配置中定义。

   这行代码的作用是告诉 CMake，任何依赖 `onnxruntime::onnxruntime` 库的目标都应该包含 `${onnxruntime_INCLUDE_DIRS}` 指定的头文件目录。

3. **`message("WIN32: ${WIN32}")`**
   - `message` 命令用于在 CMake 配置过程中输出信息。这里输出了 `WIN32` 变量的值。
   - `${WIN32}` 是一个内置的 CMake 变量，它在 Windows 平台上为 `TRUE`，在其他平台上为 `FALSE`。该变量帮助区分操作系统平台。

   这行代码的作用是打印出 `WIN32` 变量的值，用来确认当前操作系统平台是否是 Windows。

4. **`if(WIN32)`**
   - `if(WIN32)` 判断条件，检查当前平台是否是 Windows。
   - 如果是在 Windows 上构建，则会进入 `if` 分支。

   下面的代码会根据平台的不同，设置不同的库属性。

5. **`set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_IMPLIB "${onnxruntime_LIBRARY}")`**
   - 这行代码在 Windows 平台上执行，使用 `set_target_properties` 设置 `onnxruntime::onnxruntime` 库的属性。
   - `IMPORTED_IMPLIB` 属性用于指定导入库（Import Library）的路径，通常是 Windows 平台上的 `.lib` 文件。`IMPORTED_IMPLIB` 属性告诉 CMake，这个库是一个动态库（DLL），并提供了与该 DLL 相关的导入库路径。
   - `${onnxruntime_LIBRARY}` 变量包含了 ONNX Runtime 库的实现库路径，通常是 `.lib` 文件。

   这行代码的作用是在 Windows 平台上为导入的动态库（DLL）设置导入库的路径。

6. **`else()`**
   - `else()` 分支表示当平台不是 Windows 时，执行这里的代码，即 Linux、macOS 或其他操作系统。

7. **`set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_LOCATION "${onnxruntime_LIBRARY}")`**
   - 在非 Windows 平台上执行，使用 `set_target_properties` 设置 `onnxruntime::onnxruntime` 库的属性。
   - `IMPORTED_LOCATION` 属性用于指定共享库的实际位置（例如 `.so` 文件在 Linux 上，`.dylib` 文件在 macOS 上）。
   - `${onnxruntime_LIBRARY}` 变量包含了 ONNX Runtime 库的路径，通常是 `.so` 或 `.dylib` 文件。

   这行代码的作用是在 Linux/macOS 等非 Windows 平台上，指定共享库文件的路径。

8. **`endif()`**
   - `endif()` 结束了 `if(WIN32)` 判断块。

总结：

- **定义一个外部共享库目标** `onnxruntime::onnxruntime`，并指定其为已构建的库（IMPORTED）。
- **设置头文件路径**：使用 `target_include_directories` 指定 ONNX Runtime 库的头文件路径，供依赖该库的目标使用。
- **输出平台信息**：使用 `message("WIN32: ${WIN32}")` 输出当前平台是否为 Windows。
- **根据平台设置库路径**：
  - 在 **Windows** 上，设置 `IMPORTED_IMPLIB` 属性，指定导入库（.lib 文件）的路径。
  - 在 **非 Windows** 平台（如 Linux 或 macOS）上，设置 `IMPORTED_LOCATION` 属性，指定共享库文件（如 `.so` 或 `.dylib`）的路径。

总体上，代码根据平台（Windows 或非 Windows）设置了 ONNX Runtime 动态库的不同路径属性，确保在不同平台上能够正确地链接和使用 ONNX Runtime 库。









# 三 Cmakelist构建小技巧

## 设置不同构建模式下目标生成时的输出目录

[CMake生成Debug和Release目标程序时的一些配置_cmake debug-CSDN博客](https://blog.csdn.net/new9232/article/details/140567742)

```cmake
# 下述代码中的两个属性指定了不同构建模式下（Debug 和 Release）的可执行文件（或库文件）的输出目录
set_target_properties(名称 PROPERTIES
  # 名称是可执行文件，应用以下可执行文件输出目录
  # 默认是生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹下
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release
  
  # 名称是静态库文件，应用以下可执行文件输出目录
  # 默认是生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹下
  ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_SOURCE_DIR}/bin/debug
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin/release

  # 名称是动态库文件，应用以下可执行文件输出目录
  # 默认是生成根（${projectDir}\out\build\${name}）下的Debug或者Release文件夹下
  LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${PROJECT_SOURCE_DIR}/bin/debug
  LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/bin/release
)
```







# 四 参考

- 《CMake构建实战：项目开发卷》

- [全网最细的CMake教程！(强烈建议收藏) - 知乎](https://zhuanlan.zhihu.com/p/534439206)

