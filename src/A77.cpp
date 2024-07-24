#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_71_80.h"
#include <math.h>
using namespace cv;


void A77()
{
	int sigma = 10;//����
	int gamma = 1.2;//٤��
	const int kSize = 111;//����˴�С
	int kRadius = floor(kSize / 2);//����˰뾶
	float kernel[kSize][kSize];//����˾���
	float kSum = 0;
	int px = 0, py = 0;//�뾶�ڵ�����
	float theta = 0;
	float angle = 0;
	float Lambda = 10;
	float Psi = 0;
	float max = 0;
	float min=999;
	Mat gabor = Mat::zeros(kSize, kSize, CV_8UC1);
	//��������Ȩֵ
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
	printf_s("%f,%f\n",max,min);

	for (int y = 0; y < kSize; y++) 
	{
		for (int x = 0; x < kSize; x++) 
		{
			gabor.at<uchar>(y, x) = kernel[y][x] / kSum;
			gabor.at<uchar>(y, x) -= min;
			gabor.at<uchar>(y, x) /= max;
			gabor.at<uchar>(y, x) *= 255;
		}
	}

	imshow("gabor", gabor);
	waitKey(0);
	destroyAllWindows();
}