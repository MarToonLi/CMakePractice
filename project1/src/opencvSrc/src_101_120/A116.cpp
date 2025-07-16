#include <omp.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>
#include <chrono>
#include <future>
#include <algorithm>
#include <typeinfo>

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <utility>

using namespace std;
using namespace cv;
using namespace ml;

#pragma execution_character_set("utf-8") 



/**********************************
*
练习目标：
学习C++的知识：C++的控制赋值




************************************/

namespace NA116 {

	// chapter13: 拷贝控制类
    class ResourceHolder {
    private:
        // 动态分配的数组（需手动管理）
        int* dynamicArray;
        size_t arraySize;

        // 文件句柄（需手动关闭）
        std::fstream file;

        // 智能指针（自动管理）
        std::unique_ptr<int> smartPtr;

        // STL 容器（自动管理）
        std::vector<std::string> stringVec;

        // 基本类型（无需管理）
        int basicInt;

        //? 裸指针（指向外部资源，不负责释放）还是很特殊的！
        const std::string* rawPtr;

    public:
        // 构造函数
        ResourceHolder(
            size_t size, 
            const std::string& filename, 
            int smartPtrVal, 
            const std::string* rawPtr
        )
            : dynamicArray(new int[size]), 
            arraySize(size),                                 // 
            file(filename, std::ios::out | std::ios::trunc), // 初始化了一个file对象: 
            // std::fstream file("example.txt", std::ios::in | std::ios::out);
            // 或者 std::fstream file;  file.open("example.txt", std::ios::in | std::ios::out);
            smartPtr(std::make_unique<int>(smartPtrVal)),    // 初始化智能指针：需要使用make_unique创建
            basicInt(42),                                    // 内置类型
            rawPtr(rawPtr)                                   // 顶部const，内容不可变，指针可变。
        {
            std::cout << "构造函数调用\n";
        }

        // 析构函数
        ~ResourceHolder() {
            //? 裸指针还是很特殊的！不能在析构函数中释放资源，毕竟裸指针指向的内存在类外，内容资源的所有权不是类。
            delete[] dynamicArray;  // 释放动态数组
            if (file.is_open()) {
                file.close();       // 关闭文件
            }
            std::cout << "析构函数调用\n";
        }

        // 拷贝构造函数（深拷贝）
        //? 触发场景：目标对象还没有初始化；
        //? 函数签名：ClassName(const ClassName& other)
        //? 返回值：无
        ResourceHolder(const ResourceHolder& other)
            : dynamicArray(new int[other.arraySize]), //? 只能先根据已存在对象的数组长度分配内存，再深度拷贝内容构造动态数组
            arraySize(other.arraySize),
            file("copy_" + std::to_string(rand() % 100) + ".txt", std::ios::out | std::ios::trunc),  // other似乎没有提供相关信息
            smartPtr(other.smartPtr ? std::make_unique<int>(*other.smartPtr) : nullptr),
            // 如果smartPtr(other.smartPtr ? other.smartPtr : nullptr)， 会违反unique_ptr只能由一个对象拥有的设计原则
            // 如果smartPtr是sharedptr则可以直接复制
            /*
            std::unique_ptr<int> smartPtr = std::make_unique<int>(42);
            int value = *smartPtr;  // value = 42
            **/
            stringVec(other.stringVec),
            basicInt(other.basicInt),
            rawPtr(other.rawPtr) 
        {
            std::copy(other.dynamicArray, other.dynamicArray + arraySize, dynamicArray);
            std::cout << "拷贝构造函数调用\n";
        }

        // 拷贝赋值运算符（深拷贝）
        //? 触发场景：目标对象已经被初始化；使用已存在对象来赋值目标对象
        //? 函数签名：ClassName& operater=(const ClassName& other)
        //? 返回值：*this
        ResourceHolder& operator=(const ResourceHolder& other) {
            if (this != &other) {
                // 先释放当前资源
                delete[] dynamicArray;            
                if (file.is_open()) file.close();

                // 再深拷贝
                dynamicArray = new int[other.arraySize];
                arraySize = other.arraySize;
                std::copy(other.dynamicArray, other.dynamicArray + arraySize, dynamicArray);

                file.open("copy_" + std::to_string(rand() % 100) + ".txt", std::ios::out | std::ios::trunc);
                smartPtr = other.smartPtr ? std::make_unique<int>(*other.smartPtr) : nullptr;
                stringVec = other.stringVec;
                basicInt = other.basicInt;
                rawPtr = other.rawPtr;
            }
            std::cout << "拷贝赋值运算符调用\n";
            return *this;
        }

        // 移动构造函数（转移资源）
        //? 触发场景：目标对象还没有初始化；
        //? 函数签名：ClassName(const ClassName& other)
        //? 实际意义："窃取"另一个对象（通常是临时对象或即将被销毁的对象）的资源，避免深拷贝
        //? 调用方式：MyClass obj1;  MyClass obj2 = std::move(obj1);
        //? 返回值：无
        ResourceHolder(ResourceHolder&& other) noexcept           //? 移动操作应尽量标记为 noexcept： 容器优化依赖的需要
            : dynamicArray(other.dynamicArray),
            arraySize(other.arraySize),
            file(std::move(other.file)),          // 移动文件句柄
            smartPtr(std::move(other.smartPtr)),  // 移动智能指针
            stringVec(std::move(other.stringVec)),  // 移动容器
            basicInt(other.basicInt),
            rawPtr(other.rawPtr) 
        {
            other.dynamicArray = nullptr;           // 置空源对象的指针
            other.arraySize = 0;                    // 其他成员变量会触发各自的值初始化
            std::cout << "移动构造函数调用\n";
        }
        //? 一般地，new操作可能会因为系统内存不足或者指针为空，而报错，因此移动构造函数中应该避免在存在new相关代码。

        // 移动赋值运算符（转移资源）
        //? 触发场景：目标对象已经被初始化；使用已存在对象来赋值目标对象
        //? 函数签名：ClassName& operater=(const ClassName& other)
        //? 实际意义: 用于已存在的对象从另一个对象（通常是右值）"窃取"资源，避免深拷贝
        //? 调用方式：MyClass obj1, obj2;  obj2 = std::move(obj1);
        //? 返回值：*this
        ResourceHolder& operator=(ResourceHolder&& other) noexcept {
            if (this != &other) {
                // 释放当前资源
                delete[] dynamicArray;
                if (file.is_open()) file.close();

                // 转移资源
                dynamicArray = other.dynamicArray;
                arraySize = other.arraySize;
                file = std::move(other.file);
                smartPtr = std::move(other.smartPtr);
                stringVec = std::move(other.stringVec);
                basicInt = other.basicInt;
                rawPtr = other.rawPtr;

                // 置空源对象
                other.dynamicArray = nullptr;
                other.arraySize = 0;
            }
            std::cout << "移动赋值运算符调用\n";
            return *this;
        }

        // 打印资源信息（用于调试）
        void print() const {
            std::cout << "dynamicArray: " << (dynamicArray ? "有效" : "null") << "\n";
            std::cout << "file: " << (file.is_open() ? "打开" : "关闭") << "\n";
            std::cout << "smartPtr: " << (smartPtr ? std::to_string(*smartPtr) : "null") << "\n";
            std::cout << "stringVec.size(): " << stringVec.size() << "\n";
            std::cout << "basicInt: " << basicInt << "\n";
            std::cout << "rawPtr: " << (rawPtr ? *rawPtr : "null") << "\n";
        }
    };

    int chapter13_main() {
        std::string externalStr = "外部字符串";

        // 测试构造函数
        ResourceHolder rh1(5, "test1.txt", 100, &externalStr);
        rh1.print();

        // 测试拷贝构造
        ResourceHolder rh2 = rh1;
        rh2.print();

        // 测试拷贝赋值
        ResourceHolder rh3(1, "test2.txt", 200, nullptr);
        rh3 = rh1;
        rh3.print();

        // 测试移动构造
        ResourceHolder rh4 = std::move(rh1);
        rh4.print();
        rh1.print();  // rh1 的资源已被转移

        // 测试移动赋值
        ResourceHolder rh5(1, "test3.txt", 300, nullptr);
        rh5 = std::move(rh2);
        rh5.print();
        rh2.print();  // rh2 的资源已被转移

        return 0;
    }




	void A116_solver()
	{
		LOGD("ss");

        chapter13_main();


		return;
	}
}
