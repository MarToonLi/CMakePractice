#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_101_120.h"

using namespace cv;



bool transformDefectAreaPerspective(const cv::Mat & imgSrc3, const cv::Rect & bottom_rect, const int delta_x, const int delta_y, const cv::Rect & defect_rect, cv::Rect & pt_countour_rect)
{
	cv::Rect bbr = bottom_rect;
	cv::Point2f a_p1 = cv::Point2f(bbr.x, bbr.y);
	cv::Point2f a_p2 = cv::Point2f(bbr.x + bbr.width, bbr.y);
	cv::Point2f a_p3 = cv::Point2f(bbr.x + bbr.width - delta_x, bbr.y + bbr.height);
	cv::Point2f a_p4 = cv::Point2f(bbr.x + delta_x, bbr.y + bbr.height);
	std::vector<cv::Point2f> a_p_lst = { a_p1, a_p2, a_p3, a_p4 };

	cv::Point2f b_p1 = cv::Point2f(bbr.x, bbr.y);
	cv::Point2f b_p2 = cv::Point2f(bbr.x + bbr.width, bbr.y);
	cv::Point2f b_p3 = cv::Point2f(bbr.x + bbr.width, bbr.y + bbr.height + delta_y);
	cv::Point2f b_p4 = cv::Point2f(bbr.x, bbr.y + bbr.height + delta_y);
	std::vector<cv::Point2f> b_p_lst = { b_p1, b_p2, b_p3, b_p4 };

	cv::Mat A2 = cv::getPerspectiveTransform(a_p_lst, b_p_lst);

	// 整体检测
	cv::Mat result_imgSrc3;
	cv::warpPerspective(imgSrc3, result_imgSrc3, A2, imgSrc3.size(), INTER_LINEAR);

	// 构建样本
	cv::Mat imgSrc31 = cv::Mat::zeros(imgSrc3.size(), CV_8UC1);
	cv::rectangle(imgSrc31, defect_rect, cv::Scalar(255), -1);

	// 局部测试
	cv::Mat result_imgSrc31;
	cv::warpPerspective(imgSrc31, result_imgSrc31, A2, imgSrc31.size(), INTER_LINEAR);

	// 计算新区域的面积和对角线长度
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(result_imgSrc31, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	if (contours.size() >= 1) {
		pt_countour_rect = cv::boundingRect(contours[0]);
		return true;
	}
	else {
		// LOGE;contours.size()
		return false;
	}
}




void A104()
{


	//图形宽高
	printf_s("//基于角度将图像进行透视变换\n");

	cv::Mat imgSrc = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\A104.png");


	int w = imgSrc.cols;
	int h = imgSrc.rows;

	cv::Point center(static_cast<int>(w / 2), static_cast<int>(h / 2));
	double angle = 45;
	double scale = 1.0;
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, scale);

	cv::Mat rotated_image;

	cv::warpAffine(imgSrc, rotated_image, rot, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());



	//原坐标
	//cv::Point2f src[] = { cv::Point2f(839,0),cv::Point2f(1590,0),cv::Point2f(1595,582),cv::Point2f(830,590) };
	//cv::Point2f dst[] = { cv::Point2f(1150,0),cv::Point2f(1250,0),cv::Point2f(1595,582),cv::Point2f(830,590) };

	cv::Point2f src[] = { cv::Point2f(839,0 - 160),cv::Point2f(1590,0 - 160),cv::Point2f(1595,582 - 160),cv::Point2f(830,590 - 160) };
	cv::Point2f dst[] = { cv::Point2f(1150,0 - 160),cv::Point2f(1250,0 - 160),cv::Point2f(1595,582 - 160),cv::Point2f(830,590 - 160) };

	//经过投影变换后的坐标

	cv::Mat A = getPerspectiveTransform(src, dst);
	cv::Mat per_image;

	cv::warpPerspective(imgSrc, per_image, A, imgSrc.size());



	cv::Mat imgSrc2 = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\A105.png");

	cv::Mat mask1 = cv::Mat::zeros(imgSrc2.size(), CV_8UC1);
	cv::Rect rect1 = cv::Rect(cv::Point(0, 500), cv::Point(2200, 1650));
	cv::rectangle(mask1, rect1, cv::Scalar(255), -1);

	cv::Mat mask_img;
	imgSrc2.copyTo(mask_img, mask1);



	cv::Mat imgSrc3 = cv::Mat::zeros(imgSrc2.size(), CV_8UC1);
	cv::Rect moni_area_rect = cv::Rect(cv::Point(1141, 443),cv::Point(1712, 650) );
	cv::rectangle(imgSrc3, moni_area_rect, cv::Scalar(255), -1);



	// 构建一个转换矩阵
	//! 1. 定位boundingrect，作为b_rect; 而

	cv::Rect bbr = cv::Rect(cv::Point(1141, 443), cv::Point(1712, 630)); 
	int delta_x = 30; 
	int delta_y = 50;
	cv::Rect moni_area_rect31 = cv::Rect(cv::Point(1300, 500), cv::Point(1500, 600));
	cv::Rect pt_countour_rect;




	return;
}


