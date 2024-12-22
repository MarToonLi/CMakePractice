#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"

using namespace cv;
void A1_P(void)
{
	//图形宽高
	printf_s("//读取图像，然后将{RGB}通道替换成{BGR}通道\n");

	Mat imgSrc = imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\img1.png");
	int imgWidth = imgSrc.cols;
	int imgHeight = imgSrc.rows;

	//构建零图片矩阵,CV_8UC3:uchar 8bit depth:3
	//遍历图片：从左到右，从上到下
	//Mat.at:用于获取图像矩阵某点的值或改变某点的值。[0][1][2]三个通道
	//typedef Vec<uchar,3> Vec3b;0-255

	cv::Mat imgOut = cv::Mat::zeros(imgSrc.size(), CV_8UC3);

	for (int row = 0; row < imgHeight; row++)
	{
		for (int col = 0; col < imgWidth; col++)
		{
			imgOut.at<cv::Vec3b>(row, col)[0] = imgSrc.at<cv::Vec3b>(row, col)[2];
			imgOut.at<cv::Vec3b>(row, col)[1] = imgSrc.at<cv::Vec3b>(row, col)[1];
			imgOut.at<cv::Vec3b>(row, col)[2] = imgSrc.at<cv::Vec3b>(row, col)[0];
		}
	}


	return;
}


