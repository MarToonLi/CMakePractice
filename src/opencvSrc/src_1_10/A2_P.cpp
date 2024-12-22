#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"
using namespace cv;
void A2_P(void)
{
	printf_s("计算图片灰度");
	Mat imgSrc = imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\img1.png");
	int imgWidth = imgSrc.cols;
	int imgHeight = imgSrc.rows;
	
	//GRAY=b0.114 + g0.587 + r0.299 



	cv::Mat imgOut = cv::Mat::zeros(imgSrc.size(), CV_8UC1);

	for (int row = 0; row < imgHeight; row++)
	{
		for (int col = 0; col < imgWidth; col++)
		{
			imgOut.at<uchar>(row, col) = 0.114 * (float)imgSrc.at<cv::Vec3b>(row, col)[0] 
				+ 0.587 * (float)imgSrc.at<cv::Vec3b>(row, col)[1]
				+ 0.299 * (float)imgSrc.at<cv::Vec3b>(row, col)[2];
		}
	}


	return;
}