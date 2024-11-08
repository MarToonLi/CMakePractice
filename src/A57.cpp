﻿#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_51_60.h"
#include <math.h>
#include <time.h>

using namespace cv;

void A57(Mat img1, Mat img2)
{
	int imgHeight = img1.rows;
	int imgWeight = img1.cols;
	int channel = img1.channels();
	printf_s("channel:%d\n", channel);

	int imgh = img2.rows;
	int imgw = img2.cols;
	float avg1 = 0;
	float avg2 = 0;

	//子图平均灰度
	for (int y = 0; y < imgh; ++y)
		for (int x = 0; x < imgw; ++x)
			for (int c = 0; c < channel; ++c)
				avg2 += img2.at<Vec3b>(y, x)[c];
	avg2 /= (imgh * imgw);

	printf_s("avgGray2:%f\n", avg2);
	int val1 = 0, val2 = 0;
	float ZNCC = -2;
	int _x = 0, _y = 0;//匹配图片坐标
	int X = 0, Y = 0;//原图匹配坐标
	float val = 0;
	bool flag = false;
	int S = 0, T = 0;

	//归一化积相关
	//每次滑动一个像素
	for (int y = 0; y < imgHeight - imgh; ++y)
	{
		for (int x = 0; x < imgWeight - imgw; ++x)
		{
			S = 0;
			T = 0;
			val = 0;
			float zncc = 0;
			avg1 = 0;
			Mat temp = img1.clone();
			Rect rect(x, y, imgw, imgh);//左上坐标（x,y）和矩形的长(x)宽(y)
			cv::rectangle(temp, rect, Scalar(0, 0, 255), 1, LINE_AA, 0);
			cv::imshow("imgGray1", temp);
			cv::waitKey(5);

			//对应区域平均灰度
			for (int _y = 0; _y < imgh; ++_y)
				for (int _x = 0; _x < imgw; ++_x)
					for (int c = 0; c < channel; ++c)
						avg1 += img1.at<Vec3b>(y + _y, x + _x)[c];
			avg1 /= (imgh * imgw);

			//在匹配图片范围内求NCC
			for (int _y = 0; _y < imgh; ++_y)
			{
				for (int _x = 0; _x < imgw; ++_x)
				{
					for (int c = 0; c < channel; ++c)
					{
						val1 = img1.at<Vec3b>(y + _y, x + _x)[c];
						val2 = img2.at<Vec3b>(_y, _x)[c];
						S += pow((val1 - avg1), 2);
						T += pow((val2 - avg2), 2);
						val += abs(val1 - avg1) * abs(val2 - avg2);
					}
				}
			}
			zncc = val / (sqrt(S) * sqrt(T));
			printf_s("y:%d,x:%d ZNCC:%f\n", Y, X, zncc);
			if (zncc > ZNCC)
			{
				ZNCC = zncc;
				Y = y;
				X = x;
				if (ZNCC == 1)
					flag = true;
			}
			if (flag)
				break;
		}
		if (flag)
			break;
	}

	Rect rect(X, Y, imgw, imgh);//左上坐标（x,y）和矩形的长(x)宽(y)
	cv::rectangle(img1, rect, Scalar(0, 0, 255), 1, LINE_8, 0);

	printf_s("匹配坐标：y:%d,x:%d\n NCC:%f\n", Y, X, ZNCC);

	cv::imshow("img1", img1);
	cv::imshow("img2", img2);

	cv::waitKey(0);
	cv::destroyAllWindows();
}