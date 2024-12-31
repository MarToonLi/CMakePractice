#pragma once
#include <opencv2/opencv.hpp>

#include "solver_export.h"     //? 引用导出头文件，以使用num_recognizer_EXPORTS等宏定义，为动态库导出符号

#include "logger.h"
#ifndef _A_101_120_            //!? 头文件保护，防止同一个头文件在多个源文件中被重复包含，避免重复导入、执行效率影响。
#define _A_101_120_

#define PI  atan(1) * 4

using namespace cv;

void A101();
void A102();
void A103();
void A104();
void A105();

namespace NA106 {
	void A106();
}

namespace NA107 {
	void A107();
}

namespace NA108 {
	SOLVER_EXPORT void A108_solver();
}


#endif 
