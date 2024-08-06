#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_101_120.h"

using namespace cv;




void A102()
{
	//图形宽高
	printf_s("//将一个不规则图像的轮廓使用findcontour以外的方法呈现\n");

	//cv::Mat imgSrc_1 = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\A101_2_1.png");
	cv::Mat imgSrc_1 = cv::imread("D:/Myself/MachineVision/resources/HaiLun-Apple/3-5/2024_08_02_15_34_22-10-yijiao_shangxiaxian/DA2951215/ori/8__DA2951215.png");

	cv::Mat imgSrc_2 = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\A101_2_2.png");

	//! 1 先根据白色图像获取到横截面区域的蒙版
	// 1 遍历每行，收集符合要求的行：包含两种跳点，一旦找到从白到黑的跳点，则取消运行
	// 2 图像需要进行预处理：阈值分割；开运算，去除微小噪声的影响；
	
	std::vector<cv::Mat> img_BGR;
	cv::split(imgSrc_1, img_BGR);


	cv::Mat test_img = img_BGR[0];
	cv::Mat _tailmask;
	cv::threshold(test_img, _tailmask, 100, 255, cv::THRESH_BINARY);


	cv::Mat main_image = test_img.clone();
	cv::Mat cyan_image = cv::Mat(test_img.size(), CV_8UC3);
	cyan_image = cv::Scalar(128, 128, 0);

	cv::Mat _erode_tailmask, _ring_tailmask;
	cv::morphologyEx(_tailmask, _erode_tailmask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
	_ring_tailmask = _tailmask - _erode_tailmask;  // 获取环框mask

	cyan_image.copyTo(test_img, _ring_tailmask);

	return;
}


