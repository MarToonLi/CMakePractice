#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_21_30.h"

using namespace cv;
using namespace std;


// ����ͼ��ߴ糣��
const int IMAGE_HEIGHT = 256;
const int IMAGE_WIDTH = 256;

// ����ͼ��ľ�ֵ�ͱ�׼��
void calculateMeanAndStdDev(const vector<vector<int>>& image, double& mean, double& stdDev) {
	int totalPixels = IMAGE_HEIGHT * IMAGE_WIDTH;
	double sum = 0;
	double sumSquare = 0;

	// ��������ֵ�ܺͺ��ܺ͵�ƽ��
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			sum += image[i][j];
			sumSquare += image[i][j] * image[i][j];
		}
	}

	// �����ֵ�ͱ�׼��
	mean = sum / totalPixels;
	double variance = (sumSquare - sum * sum / totalPixels) / totalPixels;
	stdDev = sqrt(variance);
}

// ͼ������ֵ��̬������
void normalizeToNormalDistribution(vector<vector<int>>& image) {
	double mean, stdDev;

	// Step 1: �����ֵ�ͱ�׼��
	calculateMeanAndStdDev(image, mean, stdDev);

	// Step 2: Ӧ����̬��ӳ��
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			double normalizedValue = (image[i][j] - mean) / stdDev * 20 + 127.5; // ������Ը����������ӳ�亯��
			image[i][j] = static_cast<int>(max(0.0, min(255.0, normalizedValue))); // ȷ������ֵ�� [0, 255] ��Χ��
		}
	}
}

/*
(image[i][j] - mean) / stdDev���ⲿ�ּ��㽫ÿ������ֵ��ȥ��ֵ��Ȼ����Ա�׼���ʵ�ֱ�׼����������ת��Ϊ��׼��̬�ֲ�����ʽ����ֵΪ0����׼��Ϊ1����

* 20�������20��һ���������ӣ�������չ��������̬�����ֵ�������׼��stdDev�ܴ󣬳���20����ȷ��ӳ����ֵ��Χ�ʺ�Ŀ��ͼ��ĻҶȷ�Χ��ͨ���� [0, 255]����

+ 127.5���ⲿ����Ϊ�˽�ӳ����ֵƽ�ƣ�ʹ�þ�ֵ�ӽ���127.5����������Ŀ����ȷ����̬���������ֵ����λ�ں��ʵĻҶȷ�Χ�ڡ�

==> ���ԣ�s0 / s ��һ���������ӣ���û�ж���������ȷ�����壻����ǵ�ͨ��static_Cast ������ȡֵ��Χ��
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
				//(val - m)��������ֵ��ȥ��ֵ��ʹ��Χ�������Ļ���
				//(s0 / s * (val - m))������Ŀ���׼�����������������ֵ�ķ�Χ��
				//	+ m0������Ŀ���ֵ��������ֵ����ƽ�Ƶ�Ŀ���ֵ������
				//	(uchar)��ǿ��ת��Ϊ uchar ���ͣ�ȷ���� 0 �� 255 ��Χ�ڡ�
			}
		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}
