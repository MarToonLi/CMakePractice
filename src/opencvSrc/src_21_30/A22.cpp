#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_21_30.h"

using namespace cv;
using namespace std;


// 定义图像尺寸常量
const int IMAGE_HEIGHT = 256;
const int IMAGE_WIDTH = 256;

// 计算图像的均值和标准差
void calculateMeanAndStdDev(const vector<vector<int>>& image, double& mean, double& stdDev) {
	int totalPixels = IMAGE_HEIGHT * IMAGE_WIDTH;
	double sum = 0;
	double sumSquare = 0;

	// 计算像素值总和和总和的平方
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			sum += image[i][j];
			sumSquare += image[i][j] * image[i][j];
		}
	}

	// 计算均值和标准差
	mean = sum / totalPixels;
	double variance = (sumSquare - sum * sum / totalPixels) / totalPixels;
	stdDev = sqrt(variance);
}

// 图像像素值正态化处理
void normalizeToNormalDistribution(vector<vector<int>>& image) {
	double mean, stdDev;

	// Step 1: 计算均值和标准差
	calculateMeanAndStdDev(image, mean, stdDev);

	// Step 2: 应用正态化映射
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			double normalizedValue = (image[i][j] - mean) / stdDev * 20 + 127.5; // 这里可以根据需求调整映射函数
			image[i][j] = static_cast<int>(max(0.0, min(255.0, normalizedValue))); // 确保像素值在 [0, 255] 范围内
		}
	}
}

/*
(image[i][j] - mean) / stdDev：这部分计算将每个像素值减去均值，然后除以标准差，以实现标准化（将数据转换为标准正态分布的形式，均值为0，标准差为1）。

* 20：这里的20是一个比例因子，用于扩展或收缩正态化后的值域。如果标准差stdDev很大，乘以20可以确保映射后的值范围适合目标图像的灰度范围（通常是 [0, 255]）。

+ 127.5：这部分是为了将映射后的值平移，使得均值接近于127.5。这样做的目的是确保正态化后的像素值大致位于合适的灰度范围内。

==> 所以，s0 / s 是一个比例因子，且没有多少理论正确的意义；最后还是得通过static_Cast 来控制取值范围。
*/



void A22(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);

	int m0= 128;
	int s0= 52;

	double m, s;
	double sum = 0.0, squaredSum = 0.0;
	double val;

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				val = (double)imgSrc.at<Vec3b>(y, x)[c];
				sum += val;
				squaredSum += (val * val);
			}
		}
	}


	m = sum / (imgHeight * imgWidth * channel);
	s = sqrt(squaredSum / (imgHeight * imgWidth * channel) - (m * m));

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				val = imgSrc.at<Vec3b>(y,x)[c];
				imgOut.at<Vec3b>(y,x)[c]=(uchar)(s0 / s * (val - m) + m0);
				//(val - m)：将像素值减去均值，使其围绕零中心化。
				//(s0 / s * (val - m))：乘以目标标准差比例，以缩放像素值的范围。
				//	+ m0：加上目标均值，将像素值重新平移到目标均值附近。
				//	(uchar)：强制转换为 uchar 类型，确保在 0 到 255 范围内。
			}
		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}
