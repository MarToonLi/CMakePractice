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


using namespace std;
using namespace cv;

/*
功能：

1. 获取memory
1. C++常见犯错点



*/

#pragma execution_character_set("utf-8") 


namespace NA114 {
	/*
	// 获取memory
	#include <windows.h>  
	#include <psapi.h>
	#include <stdio.h>
	**/

	// ---- get memory info ---- //
	bool getMemoryInfo(double& decimal_total, double& decimal_avl)
	{
		try {
			const int kMaxInfoBuffer = 256;
			int GBYTES = 1073741824;
			int MBYTES = 1048576;
			int KBYTES = 1024;
			float DKBYTES = 1024.0;

			std::string memory_info;
			MEMORYSTATUSEX statusex;
			statusex.dwLength = sizeof(statusex);

			if (GlobalMemoryStatusEx(&statusex))
			{
				unsigned long long total = 0, remain_total = 0, avl = 0, remain_avl = 0;
				remain_total = statusex.ullTotalPhys % GBYTES;
				total = statusex.ullTotalPhys / GBYTES;
				avl = statusex.ullAvailPhys / GBYTES;
				remain_avl = statusex.ullAvailPhys % GBYTES;
				if (remain_total > 0)
					decimal_total = (remain_total / MBYTES) / DKBYTES;
				if (remain_avl > 0)
					decimal_avl = (remain_avl / MBYTES) / DKBYTES;

				decimal_total += (double)total;
				decimal_avl += (double)avl;
				char  buffer[kMaxInfoBuffer];
				sprintf_s(buffer, kMaxInfoBuffer, "total %.2f GB (%.2f GB available)", decimal_total, decimal_avl);
				memory_info.append(buffer);
			}
			std::cout << memory_info << std::endl;
			return true;

		}
		catch (const cv::Exception& e)
		{
			LOGE("E1 {}; -1000", e.what());
		}
		catch (std::exception& e)
		{
			LOGE("E2 {}; -1000", e.what());
		}
		catch (...)
		{
			LOGE("E3 {}; -1000");
		}
		return false;

	}

	bool getCurThreadMemoryInfo(double& curThreadMem)
	{
		PROCESS_MEMORY_COUNTERS pmc;
		if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
		{
			curThreadMem = pmc.WorkingSetSize / 1024 / 1024;
			LOGD("当前进程占用内存大小为：{} MB\n", curThreadMem);
			return true;
		}
		else {
			LOGD("fialed to fetch curThreadMem.");
			return false;
		}
	}


	//https://www.bilibili.com/video/BV1rC4y1a7WR/?spm_id_from=333.337.search-card.all.click&vd_source=10e925901643d8014b3fac9e1258fe32

	// 1 等号的赋值操作，而不是比较操作
	void test1()
	{
		LOGD("test1");
		int x = 0;
		if (x = 2) {
			LOGD("x1: {};", x);  // 此时会将x赋值为2！
		}
		else {
			LOGD("x2: {};", x);
		}
	}

	// 2. 形参和实参的问题（int arr[] = {0} 中arr虽然是指针变量，但是它属于常量指针，不允许被修改）
	void foo(int* p) {
		p++;
	}

	// 3. ？
	void test2()
	{
		LOGD("test2");
		int arr[] = { 0 };   // 定义数组arr, 它是一个指向int的指针(int *), 指向arr数组的第一个元素


		//? 错误示范：修改值
		LOGD("arr: {};", *arr);   // *arr表示第一个元素的值
		foo(arr);                 // 传入arr，指针值传递，p是arr的副本；p++使得p指向arr[1]，产生越界，但是不影响arr本身；
		LOGD("arr[0]: {};", arr[0]);
		LOGD("arr: {};", *arr);


		//? task2: 修改int[]类型数组中首个字符的值
		int arr2[] = { 0 };   // 定义数组arr, 它是一个指向int的指针(int *), 指向arr数组的第一个元素
		int* p2 = arr2;
		(*p2)++;
		LOGD("arr2.*arr2: {};", *arr2);  // 1
		LOGD("arr2.*p2  : {};", *p2);    // 1


		//? task1: 修改int[]类型数组中首个字符的指针的值
		int arr3[] = { 0 };       // 定义数组arr, 它是一个指向int的指针(int *), 指向arr数组的第一个元素
		int* p3 = arr3;           // p3表示int[]类型数组首个元素的指针值
		int** p3_address = &p3;   // p3_address 表示int*指针p3的指针
		//(*p3_address)++;          // *p3_address表示int*指针p3的值(地址)

		/*** 结论1: 查看编译时的变量类型：typeid + decltype ***/
		// #include <typeinfo>
		// decltype: ​在编译时推导表达式的类型,但不实际计算表达式的值;
		// decltype(expression) var;  // 声明一个变量 var，其类型与 expression 相同
		std::cout << "arr3 的类型: " << typeid(decltype(arr3)).name() << std::endl;                // int [1]
		std::cout << "p3 的类型: " << typeid(decltype(p3)).name() << std::endl;                    // int * __ptr64
		std::cout << "(void*)arr3 的类型: " << typeid(decltype((void*)arr3)).name() << std::endl;  // void * __ptr64


		/*** 结论2: arr3保存了类型信息，能够通过*实现指针算术计算；(void*)arr3常用语spdlog的打印或者传递到泛型函数中作为借口 ***/
		// 000000E7E26FEE74 和 0xe7e26fee74 是同一个内存地址的不同表示形式，不同点在格式和补齐方式
		LOGD("arr3.0.address: {}", (void*)arr3);
		//LOGD("arr3.0.address: {}", arr3);  //! 不可输出
	


		/*** 结论3: 一个指针的值可以由指针变量名输出获取，或者指向该指针的另一个指针的解引用&获取。 ***/
		std::cout << "arr3.0.val: " << arr3[0] << std::endl;   // arr首个元素的值
		std::cout << "arr3.0.val: " << *p3 << std::endl;       // arr首个元素的值
		std::cout << "arr3.0.address: " << arr3 << std::endl;          // arr首个元素的值的地址
		std::cout << "arr3.p3.val: " << p3 << std::endl;               // p3的值
		std::cout << "arr3.p3.address: " << &p3 << std::endl;          //! p3的地址，取值符可以获取
		std::cout << "arr3.p3.address: " << p3_address << std::endl;   //! p3的地址，p3_address的值
		std::cout << "arr3.p3_address.val: " << p3_address << std::endl;  // p3_address的值是p3的值
		std::cout << "arr3.p3_address.address: " << &p3_address << std::endl; // p3_address的地址，取值符可以获取


		// & 取址运算符  * 解引用运算符（*只能用于指针类型！）
	}

	// 4. ?
	void test3()
	{
		// 我们需要什么？
		std::string s1 = "string";
		std::vector<int> v1;
		v1.push_back(0);

		std::string::size_type s_i = 0;
		std::vector<int>::size_type v_i = 0;  // unsigned __int64类型;

		LOGD("s1.size.type: {}; vector.size.type: {};", typeid(decltype(s1.size())).name(), typeid(decltype(v1.size())).name());
		LOGD("s_i.size.type: {}; v_i.size.type: {};", typeid(decltype(s_i)).name(), typeid(decltype(v_i)).name());
	}

	
	void A114_solver()
	{
		LOGD("ss");

		double curThreadMem = -1;
		double decimal_total = -1;
		double decimal_avl = -1;

		bool temp1 = getMemoryInfo(decimal_total, decimal_avl);
		bool temp2 = getCurThreadMemoryInfo(curThreadMem);

		
		//local_meaner = IncrementalMean();

		test1();

		test2();

		test3();




		return;
	}
}
