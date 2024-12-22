﻿#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_61_70.h"
#include <math.h>

using namespace cv;

void A65(Mat img)
{
	int H = img.rows;
	int W = img.cols;
	int channel = img.channels();
	int val = 0;
	//灰度二值化
	Mat imgGray = Mat::zeros(H, W, CV_8UC1);
	Mat imgBin = Mat::zeros(H, W, CV_8UC1);

	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.114 * (float)img.at<Vec3b>(y, x)[2]
				+ 0.587 * (float)img.at<Vec3b>(y, x)[1]
				+ 0.299 * (float)img.at<Vec3b>(y, x)[0];
		}
	}

	const int grayScale = 256;//灰度值

	int pixelCount[grayScale] = { 0 };//灰度直方图
	float pixelPro[grayScale] = { 0 };//各个灰度值占总体的比例

	double w0, w1;//背景/前景像素占比
	double u0, u1;//前景/背景平均灰度值
	double p0, p1;
	double g = 0;//类间方差

	double max_g = 0;//最大类间方差
	double good_k = 0;//最优阈值
	int pixelSum = H * W;//总像素值

	//统计图片中各个灰度值的个数
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			int val = imgGray.at<uchar>(y, x);
			pixelCount[val]++;
		}
	}

	//统计图片中各个灰度值所占的比例
	for (int i = 0; i < grayScale; ++i)
	{
		pixelPro[i] = 1.0 * pixelCount[i] / pixelSum;
	}

	//k:暂定阈值(0-255)
	for (int k = 0; k < grayScale; ++k)
	{
		w0 = w1 = u0 = u1 = g = 0;
		p0 = p1 = 0;
		//前景，背景区分 [0-k][k+1-255]
		for (int i = 0; i < grayScale; ++i)
		{
			//如果当前像素值小于阈值k则属于背景，反之属于前景
			if (i <= k)
			{
				//计算背景像素占比
				w0 += pixelPro[i];
				//计算当前灰度值发生的概率:灰度值*灰度值发生的概率
				p0 += (i * pixelPro[i]);

			}
			else
			{
				//计算背景像素占比
				w1 += pixelPro[i];
				p1 += (i * pixelPro[i]);
			}
		}
		//计算平均灰度值：p0/w0
		u0 = p0 / w0;
		u1 = p1 / w1;
		//计算类内方差
		g = (float)(w0 * w1 * pow((u0 - u1), 2));
		if (g > max_g)
		{
			max_g = g;
			good_k = k;

		}
	}

	printf_s("good k;%f\n", good_k);

	//翻转颜色：0作为线，将1作为背景
	Mat imgRe = Mat::zeros(H, W, CV_8UC1);

	//取得最好的k值，以k值作为阈值进行二值化
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			if (imgGray.at<uchar>(y, x) > good_k)
			{
				imgBin.at<uchar>(y, x) = 1;
				imgRe.at<uchar>(y, x) = 0;
			}
			else
			{
				imgBin.at<uchar>(y, x) = 0;
				imgRe.at<uchar>(y, x) = 1;
			}
		}
	}

	int flag = 0;
	bool find = true;
	while (find)
	{
		find = false;
		for (int y = 1; y < H - 1; ++y)
		{
			for (int x = 1; x < W - 1; ++x)
			{
				//排除白色
				if (imgRe.at<uchar>(y, x) == 1)
					continue;
				flag = 0;
				if ((imgRe.at<uchar>(y - 1, x + 1) - imgRe.at<uchar>(y - 1, x)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y, x + 1) - imgRe.at<uchar>(y - 1, x + 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y + 1, x + 1) - imgRe.at<uchar>(y, x + 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y + 1, x) - imgRe.at<uchar>(y + 1, x + 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y + 1, x - 1) - imgRe.at<uchar>(y + 1, x)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y, x - 1) - imgRe.at<uchar>(y + 1, x - 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y - 1, x - 1) - imgRe.at<uchar>(y, x - 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y - 1, x) - imgRe.at<uchar>(y - 1, x - 1)) == 1)
					flag += 1;
				if (flag != 1)
					continue;

				int sum = 0;
				for (int _y = -1; _y < 2; ++_y)
				{
					for (int _x = -1; _x < 2; ++_x)
					{
						if (_y == 0 && _x == 0)
							continue;
						sum += imgRe.at<uchar>(y + _y, x + _x);
					}
				}
				if (sum < 2 || sum > 6)
					continue;
				//右半圈
				if (imgRe.at<uchar>(y - 1, x) + imgRe.at<uchar>(y, x + 1) + imgRe.at<uchar>(y + 1, x) < 1)
					continue;
				// 左半圈
				if (imgRe.at<uchar>(y, x + 1) + imgRe.at<uchar>(y + 1, x) + imgRe.at<uchar>(y, x - 1) < 1)
					continue;
				imgBin.at<uchar>(y, x) = 0;
				imgRe.at<uchar>(y, x) = 1;
				find = true;
			}
		}

		for (int y = 1; y < H - 1; ++y)
		{
			for (int x = 1; x < W - 1; ++x)
			{
				//排除白色
				if (imgRe.at<uchar>(y, x) == 1)
					continue;
				flag = 0;
				if ((imgRe.at<uchar>(y - 1, x + 1) - imgRe.at<uchar>(y - 1, x)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y, x + 1) - imgRe.at<uchar>(y - 1, x + 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y + 1, x + 1) - imgRe.at<uchar>(y, x + 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y + 1, x) - imgRe.at<uchar>(y + 1, x + 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y + 1, x - 1) - imgRe.at<uchar>(y + 1, x)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y, x - 1) - imgRe.at<uchar>(y + 1, x - 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y - 1, x - 1) - imgRe.at<uchar>(y, x - 1)) == 1)
					flag += 1;
				if ((imgRe.at<uchar>(y - 1, x) - imgRe.at<uchar>(y - 1, x - 1)) == 1)
					flag += 1;
				if (flag != 1)
					continue;

				int sum = 0;
				for (int _y = -1; _y < 2; ++_y)
				{
					for (int _x = -1; _x < 2; ++_x)
					{
						if (_y == 0 && _x == 0)
							continue;
						sum += imgRe.at<uchar>(y + _y, x + _x);
					}
				}
				if (sum < 2 || sum > 6)
					continue;

				//上半圈
				if (imgRe.at<uchar>(y - 1, x) + imgRe.at<uchar>(y, x + 1) + imgRe.at<uchar>(y, x - 1) < 1)
					continue;

				// 下半圈
				if (imgRe.at<uchar>(y - 1, x) + imgRe.at<uchar>(y + 1, x) + imgRe.at<uchar>(y, x - 1) < 1)
					continue;

				imgBin.at<uchar>(y, x) = 0;
				imgRe.at<uchar>(y, x) = 1;
				find = true;
			}
		}
	}
	//扩张像素
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			if (imgBin.at<uchar>(y, x) == 0)
				continue;
			imgBin.at<uchar>(y, x) = imgBin.at<uchar>(y, x) * 255;
		}
	}
	cv::imshow("img", img);
	cv::imshow("imgBin", imgBin);
	cv::waitKey(0);
	cv::destroyAllWindows();
}