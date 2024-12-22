#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <stdio.h>
#include "utils\logger.h"
#include "opencvSrc\src_101_120\A_101_120.h"
#include "onnxSrc\onnx_deploy.h"

//using namespace std;
//using namespace cv;

int main()
{
	SetConsoleOutputCP(CP_UTF8); // allow the chinese log to show.

	//LOGD("sssssssssssssss");

	std::cout << "sss" << std::endl;

	//NA106::A106();
	//NA107::A107();
	NA108::A108();


	yolo_min();

	return 1;
}


/*
每次练习问题记录
A1: BGR图像转换为RGB图像
1. cv::Vec3i;



A2: BGR图像转换成灰度图
1. imgOut.at<uchar><row, col>

*/