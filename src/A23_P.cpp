#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_21_30.h"

using namespace cv;

void A23(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);

	//S���ܵ���������Z_{ max }�����ص�����ȡֵ����������255����
	//h(z)��ʾȡֵΪz���ۻ��ֲ������� Z' = \frac{Z_{max}}{S} \ \sum\limits_{i=0}^z\ h(i) 

	double ZMax = 255;
	double hist[255];
	double S = imgHeight * imgWidth * channel;
	int val;
	double histSum = 0;

	//��ʼ��
	for (int i = 0; i < 255; ++i)
		hist[i] = 0;

	//ͳ�����طֲ�
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
	//ֱ��ͼ����

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				val = imgSrc.at<Vec3b>(y, x)[c];
				histSum = 0;
				//����0-val���������ظ����Ĺ���sum
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
