#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_101_120.h"

using namespace cv;



struct rowInfo
{
	cv::Point p1 = cv::Point(0, 0);
	cv::Point p2 = cv::Point(0, 0);
	double center_offset = -1;
	double white_len = -1;
};





void function1(cv::Mat& imgsrc)
{
	cv::Mat gray_img;
	cv::cvtColor(imgsrc, gray_img, COLOR_BGR2GRAY);

	cv::Mat thresh_img;
	cv::Mat _tmp1; cv::Mat _tmp2;
	cv::threshold(gray_img, _tmp1, 10, 0, cv::THRESH_TOZERO);
	cv::threshold(_tmp1, _tmp2, 240, 0, cv::THRESH_TOZERO_INV);
	cv::threshold(_tmp2, thresh_img, 20, 255, cv::THRESH_BINARY);

	// 100 ;  200
	cv::Mat open_img;
	cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(101, 21));
	cv::morphologyEx(thresh_img, open_img, cv::MORPH_OPEN, kernel1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(open_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


	std::vector<std::vector<cv::Point>> candidate_contours;
	std::vector<cv::Point> hull_contour;
	cv::Mat _tmp = cv::Mat::zeros(imgsrc.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); i++)
	{

		std::vector<cv::Point> cur_contour = contours[i];

		_tmp = 0;
		cv::drawContours(_tmp, contours, i, cv::Scalar(255), -1);

		int area = cv::countNonZero(_tmp);
		std::cout << area << std::endl;

		if (area > 20000 && area < 50000)
		{
			candidate_contours.push_back(cur_contour);
			hull_contour.insert(hull_contour.end(), cur_contour.begin(), cur_contour.end());
		}
	}


	std::vector<cv::Point> hull;
	std::vector<std::vector<cv::Point>> hull_wrap;
	hull_wrap.clear();
	if (hull_contour.size() != 0)
	{
		cv::convexHull(hull_contour, hull);
		hull_wrap.push_back(hull);
	}

	cv::drawContours(imgsrc, hull_wrap, 0, cv::Scalar(0, 0, 255));
}







bool function2(cv::Mat& imgsrc)
{
	cv::Mat imgSrc_1_rst;
	imgSrc_1_rst = imgsrc.clone();

	cv::Mat gray_img;
	cv::cvtColor(imgsrc, gray_img, cv::COLOR_BGR2GRAY);

	cv::Mat thresh_img;
	cv::threshold(gray_img, thresh_img, 100, 255, cv::THRESH_BINARY);

	cv::Mat open_img;
	cv::Mat ellipse_kernel_11 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	cv::morphologyEx(thresh_img, open_img, cv::MORPH_OPEN, ellipse_kernel_11);

	// 找到大概的row后，上下的距离值变得大些；


	int stop_count = 0;
	std::vector<rowInfo> finalRowsInfoVec;
	int position_center_x = int(open_img.cols / 2);
	for (int row = 0; row < open_img.rows; row++)
	{
		rowInfo final_row_info;
		std::vector<rowInfo> rowsInfoVec;
		rowInfo row_info;
		for (int col = 0; col < open_img.cols - 1; col++)
		{
			uchar z = open_img.at<uchar>(row, col);
			uchar z_next = open_img.at<uchar>(row, col + 1);

			if (z == 0 && z_next == 255)
			{
				// row_info 重置
				row_info.p1 = cv::Point(0, 0);
				row_info.p2 = cv::Point(0, 0);
				row_info.center_offset = 0;
				row_info.white_len = -1;

				// row_info 重新赋值
				row_info.p1 = cv::Point(col, row);
			}

			if (z == 255 && z_next == 0 && row_info.p1.x != 0)
			{
				// row_info 重新赋值
				row_info.p2 = cv::Point(col, row);
				row_info.center_offset = abs((int)((row_info.p2.x + row_info.p1.x) / 2) - position_center_x);
				row_info.white_len = abs(row_info.p2.x - row_info.p1.x);

				if (row_info.white_len > 250 && row_info.white_len < 850 && row_info.center_offset < 500)
				{
					rowsInfoVec.push_back(row_info);
				}
			}
		}

		if (rowsInfoVec.size() == 0)
		{
			stop_count++;
			if (stop_count > 100)
			{
				break;
			}
		}
		else if (rowsInfoVec.size() >= 1)
		{
			std::sort(rowsInfoVec.begin(), rowsInfoVec.end(), [](const rowInfo& a, const rowInfo& b) {
				return a.center_offset < b.center_offset;
				});
			final_row_info = rowsInfoVec[0];

			if (rowsInfoVec.size() >= 2)
			{
				std::cout << row << "::  " << position_center_x << std::endl;
			}

			// 更新position_center_x
			position_center_x = (position_center_x + (final_row_info.p2.x + final_row_info.p1.x) / 2) / 2;

			// 确定该行最终的final_row_info
			finalRowsInfoVec.push_back(final_row_info);

			// 可视化
			cv::line(imgSrc_1_rst, final_row_info.p1, final_row_info.p2, cv::Scalar(0, 0, 255), 10);
			//std::cout << row << "::  " << position_center_x << std::endl;
			cv::circle(imgSrc_1_rst, cv::Point(position_center_x, row), 10, cv::Scalar(255, 0, 0), -1);
		}
	}


	int roi_y = -1;
	int _count = 0;
	rowInfo roi_row_info;

	int _interval = 100;
	for (int i = _interval; i < finalRowsInfoVec.size(); i++)
	{
		rowInfo cur_row_info = finalRowsInfoVec[i];
		if (cur_row_info.white_len > 650 && cur_row_info.white_len < 800)
		{
			roi_y = finalRowsInfoVec[i - _interval].p1.y;
			roi_row_info = finalRowsInfoVec[i - _interval];
		}
		if (cur_row_info.white_len < 600)
		{
			_count++;
			if (_count > 50)
			{
				break;
			}
		}
	}
	std::cout << "roi_y : " << roi_y << std::endl;


	// 线 300 pixel； boot: 650 pixel;


	cv::Point p1(roi_row_info.p1.x - 50, roi_row_info.p1.y - 100);
	cv::Point p2(roi_row_info.p2.x + 50, roi_row_info.p1.y + 300);

	cv::Rect roi_rect = cv::Rect(p1, p2);
	cv::rectangle(imgSrc_1_rst, roi_rect, cv::Scalar(0, 255, 0), 5);

	return 1;
}





void A101()
{
	//图形宽高
	printf_s("//计算出溢胶区域的面积\n");

	//cv::Mat imgSrc_1 = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\A101_2_1.png");
	cv::Mat imgSrc_1 = cv::imread("D:/Myself/MachineVision/resources/HaiLun-Apple/3-5/2024_08_02_15_34_22-10/DA2951215/ori/8__DA2951215.png");

	cv::Mat imgSrc_2 = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\A101_2_2.png");

	//! 1 先根据白色图像获取到横截面区域的蒙版
	// 1 遍历每行，收集符合要求的行：包含两种跳点，一旦找到从白到黑的跳点，则取消运行
	// 2 图像需要进行预处理：阈值分割；开运算，去除微小噪声的影响；
	
	std::vector<cv::Mat> img_BGR;
	cv::split(imgSrc_1, img_BGR);


	function2(imgSrc_1);


	return;
}


