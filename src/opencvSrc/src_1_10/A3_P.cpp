#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"

using namespace cv;

void A3_P(int th)
{
	printf_s("把图像进行二值化");
	Mat imgSrc = imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\img1.png");
	int imgHeight = imgSrc.rows;
	int imgWeight = imgSrc.cols;
	

	cv::Mat imgOut1 = cv::Mat::zeros(imgSrc.size(), CV_8UC1);
	for (int row = 0; row < imgSrc.rows; row++)
	{
		for (int col = 0; col < imgSrc.cols; col++)
		{
			imgOut1.at<uchar>(row, col) = 0.1 * imgSrc.at<cv::Vec3b>(row, col)[0]
				+ 0.6 * imgSrc.at<cv::Vec3b>(row, col)[1] 
				+ 0.3 * imgSrc.at<cv::Vec3b>(row, col)[2];
		}
	}


	cv::Mat imgOut2 = cv::Mat::zeros(imgSrc.size(), CV_8UC1);
	for (int row = 0; row < imgSrc.rows; row++)
	{
		for (int col = 0; col < imgSrc.cols; col++)
		{
			uchar z = imgOut1.at<uchar>(row, col);
			if (z > th)
			{
				imgOut1.at<uchar>(row, col) = 255;
			}
			else
			{
				imgOut1.at<uchar>(row, col) = 0;
			}
		}
	}

	return;
}