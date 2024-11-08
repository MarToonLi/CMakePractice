#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_21_30.h"

using namespace cv;
using namespace std;

// 定义图像尺寸常量
const int IMAGE_HEIGHT = 256;
const int IMAGE_WIDTH = 256;

// 图像均衡化函数
void histogramEqualization(vector<vector<int>>& image) {
	int totalPixels = IMAGE_HEIGHT * IMAGE_WIDTH;
	vector<int> histogram(256, 0);

	// Step 1: 计算直方图
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			histogram[image[i][j]]++;
		}
	}

	// Step 2: 计算累积分布函数（CDF）
	vector<int> cdf(256, 0);
	cdf[0] = histogram[0];
	for (int i = 1; i < 256; ++i) {
		cdf[i] = cdf[i - 1] + histogram[i];
	}

	// Step 3: 应用均衡化映射
	vector<int> mapping(256, 0);
	for (int i = 0; i < 256; ++i) {
		mapping[i] = (int)((double)cdf[i] / totalPixels * 255);
	}

	// Step 4: 生成输出图像
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			image[i][j] = mapping[image[i][j]];
		}
	}
}

void A23(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);


	double ZMax = 255;
	double hist[255];
	double S = imgHeight * imgWidth * channel;
	int val;
	double histSum = 0;

	//初始化
	for (int i = 0; i < 255; ++i)
		hist[i] = 0;

	//统计像素分布
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				val = imgSrc.at<Vec3b>(y, x)[c];
				hist[val]++;
			}
		}
	}
	//直方图均衡

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				val = imgSrc.at<Vec3b>(y, x)[c];
				histSum = 0;
				//根据0-val的所有像素个数的构建sum
				for (int l = 0; l < val; l++)
				{
					histSum += hist[l];
				}
				imgOut.at<Vec3b>(y, x)[c] = (uchar)(ZMax / S * histSum);
			}
		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}
