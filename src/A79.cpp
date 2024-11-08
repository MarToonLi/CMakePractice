﻿#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_71_80.h"
#include <math.h>
using namespace cv;

int kRadius=0;

Mat GetGobar(float Angle)
{
	float kSum = 0;
	int px = 0, py = 0;//半径内的坐标

	int sigma = 10;//方差
	int gamma = 1.2;//伽马
	const int kSize = 111;//卷积核大小
	kRadius = floor(kSize / 2);//卷积核半径
	float kernel[kSize][kSize];//卷积核矩阵
	float theta = 0;
	float angle = Angle;
	float Lambda = 10;
	float Psi = 0;

	float max = 0;
	float min = 999;
	Mat gabor = Mat::zeros(kSize, kSize, CV_32FC1);
	//计算卷积核权值
	for (int y = 0; y < kSize; ++y)
	{
		for (int x = 0; x < kSize; ++x)
		{
			px = x - kRadius;
			py = y - kRadius;

			theta = angle / 180. * PI;

			float _x = cos(theta) * px + sin(theta) * py;
			float _y = sin(theta) * px + cos(theta) * py;

			kernel[y][x] = exp(-(pow(_x, 2) + pow(gamma, 2) * pow(_y, 2)) / (2 * pow(sigma, 2))) * cos(2 * PI * _x / Lambda + Psi);

			kSum += abs(kernel[y][x]);
			max = MAX(max, kernel[y][x]);
			min = MIN(min, kernel[y][x]);

		}
	}
	printf_s("%f,%f\n", max, min);

	for (int y = 0; y < kSize; y++)
	{
		for (int x = 0; x < kSize; x++)
		{
			gabor.at<float>(y, x) = kernel[y][x] / kSum;
			gabor.at<float>(y, x) -= min;
			gabor.at<float>(y, x) /= max;
			gabor.at<float>(y, x) *= 255;
		}
	}
	return gabor;
}

void A79(Mat img)
{
	cv::resize(img, img, cv::Size(300, 250));
	//灰度化
	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();
	Mat imgGray = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	//灰度化
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.2126 * (float)img.at<Vec3b>(y, x)[2] \
				+ 0.7152 * (float)img.at<Vec3b>(y, x)[1] \
				+ 0.0722 * (float)img.at<Vec3b>(y, x)[0];
		}
	}
	float angleA[] = { 0,45,90,135 };
	char name[4][5] = { "out1","out2","out3","out4" };

	Mat gaborOut[4];
	Mat imgOut[4];

	for (int i = 0; i < 4; ++i)
		gaborOut[i] = GetGobar(angleA[i]);

	Mat out = Mat::zeros(imgHeight, imgWidth, CV_32FC1);
	printf_s("%d\n", kRadius);
	for (int t = 0; t < 4; ++t)
	{
		for (int y = 0; y < imgHeight; ++y)
		{
			for (int x = 0; x < imgWidth; ++x)
			{
				float val = 0;
				for (int dy = -kRadius; dy < kRadius + 1; ++dy)
				{
					for (int dx = -kRadius; dx < kRadius + 1; ++dx)
					{
						if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
						{
							val += imgGray.at<uchar>(y + dy, x + dx) * gaborOut[t].at<float>(kRadius + dy, kRadius + dx);

						}
					}
				}
				out.at<float>(y, x) = (float)val;
			}
		}
		printf_s("ss");
		imgOut[t] = out;
	}
	for (int i = 0; i < 4; ++i)
		imshow(name[i], imgOut[i]);

	waitKey(0);
	destroyAllWindows();
}