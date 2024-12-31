#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_101_120.h"

using namespace cv;




void A103()
{


	//图形宽高
	printf_s("//将一个不规则图像的轮廓使用findcontour以外的方法呈现\n");

	cv::Mat imgSrc = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\103.png");
	
	cv::Mat _kernel_ellipse_11 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
	cv::Mat _kernel_ellipse_21 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
	cv::Mat _kernel_ellipse_31 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
	cv::Mat _kernel_ellipse_41 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(41, 41));
	cv::Mat _kernel_ellipse_51 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(51, 51));
	cv::Mat _kernel_ellipse_61 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(61, 61));
	cv::Mat _kernel_ellipse_71 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(71, 71));
	cv::Mat _kernel_ellipse_81 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(81, 81));
	cv::Mat _kernel_ellipse_91 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(91, 91));

	cv::Mat _kernel_ellipse_101 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(101, 101));
	cv::Mat _kernel_ellipse_111 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(111, 111));
	cv::Mat _kernel_ellipse_151 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(151, 151));
	cv::Mat _kernel_ellipse_201 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(201, 201));
	cv::Mat _kernel_ellipse_211 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(211, 211));



	//cv::Mat _kernel_rect_11 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	//cv::Mat _kernel_rect_21 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
	//cv::Mat _kernel_rect_31 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
	//cv::Mat _kernel_rect_41 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(41, 41));
	//cv::Mat _kernel_rect_51 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(51, 51));
	//cv::Mat _kernel_rect_61 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(61, 61));
	//cv::Mat _kernel_rect_71 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(71, 71));
	//cv::Mat _kernel_rect_81 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(81, 81));
	//cv::Mat _kernel_rect_91 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(91, 91));

	//cv::Mat _kernel_rect_101 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(101, 101));
	//cv::Mat _kernel_rect_111 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(111, 111));
	//cv::Mat _kernel_rect_151 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(151, 151));
	//cv::Mat _kernel_rect_201 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(201, 201));
	//cv::Mat _kernel_rect_211 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(211, 211));


	cv::Mat image11;
	cv::Mat image21;
	cv::Mat image31;
	cv::Mat image41;
	cv::Mat image51;
	cv::Mat image61;
	cv::Mat image71;
	cv::Mat image81;
	cv::Mat image91;

	cv::Mat image101;
	cv::Mat image111;
	cv::Mat image151;
	cv::Mat image201;
	cv::Mat image211;

	// MORPH_ELLIPSE
	cv::morphologyEx(imgSrc, image11, cv::MORPH_OPEN, _kernel_ellipse_11);  // 4
	cv::morphologyEx(imgSrc, image21, cv::MORPH_OPEN, _kernel_ellipse_21);  // 8
	cv::morphologyEx(imgSrc, image31, cv::MORPH_OPEN, _kernel_ellipse_31);  // 13
	cv::morphologyEx(imgSrc, image41, cv::MORPH_OPEN, _kernel_ellipse_41);  // 17
	cv::morphologyEx(imgSrc, image51, cv::MORPH_OPEN, _kernel_ellipse_51);  // 22
	cv::morphologyEx(imgSrc, image61, cv::MORPH_OPEN, _kernel_ellipse_61);  // 26
	cv::morphologyEx(imgSrc, image71, cv::MORPH_OPEN, _kernel_ellipse_71);  // 31
	cv::morphologyEx(imgSrc, image81, cv::MORPH_OPEN, _kernel_ellipse_81);  // 35
	cv::morphologyEx(imgSrc, image91, cv::MORPH_OPEN, _kernel_ellipse_91);  // 40

	cv::morphologyEx(imgSrc, image101, cv::MORPH_OPEN, _kernel_ellipse_101);  // 40
	cv::morphologyEx(imgSrc, image111, cv::MORPH_OPEN, _kernel_ellipse_111);  // 40
	cv::morphologyEx(imgSrc, image151, cv::MORPH_OPEN, _kernel_ellipse_151);  // 40
	cv::morphologyEx(imgSrc, image201, cv::MORPH_OPEN, _kernel_ellipse_201);  // 40
	cv::morphologyEx(imgSrc, image211, cv::MORPH_OPEN, _kernel_ellipse_211);  // 40


	// MORPH_rect
	//cv::morphologyEx(imgSrc, image11, cv::MORPH_OPEN, _kernel_rect_11);  // 4
	//cv::morphologyEx(imgSrc, image21, cv::MORPH_OPEN, _kernel_rect_21);  // 8
	//cv::morphologyEx(imgSrc, image31, cv::MORPH_OPEN, _kernel_rect_31);  // 13
	//cv::morphologyEx(imgSrc, image41, cv::MORPH_OPEN, _kernel_rect_41);  // 17
	//cv::morphologyEx(imgSrc, image51, cv::MORPH_OPEN, _kernel_rect_51);  // 22
	//cv::morphologyEx(imgSrc, image61, cv::MORPH_OPEN, _kernel_rect_61);  // 26
	//cv::morphologyEx(imgSrc, image71, cv::MORPH_OPEN, _kernel_rect_71);  // 31
	//cv::morphologyEx(imgSrc, image81, cv::MORPH_OPEN, _kernel_rect_81);  // 35
	//cv::morphologyEx(imgSrc, image91, cv::MORPH_OPEN, _kernel_rect_91);  // 40

	//cv::morphologyEx(imgSrc, image101, cv::MORPH_OPEN, _kernel_rect_101);  // 40
	//cv::morphologyEx(imgSrc, image111, cv::MORPH_OPEN, _kernel_rect_111);  // 40
	//cv::morphologyEx(imgSrc, image151, cv::MORPH_OPEN, _kernel_rect_151);  // 40
	//cv::morphologyEx(imgSrc, image201, cv::MORPH_OPEN, _kernel_rect_201);  // 40
	//cv::morphologyEx(imgSrc, image211, cv::MORPH_OPEN, _kernel_rect_211);  // 40


	return;
}


