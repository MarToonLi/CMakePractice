#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_51_60.h"
#include <math.h>
#include <time.h>
using namespace cv;

void A52(Mat img)
{
	int imgHeight = img.rows;
	int imgWeight = img.cols;
	//�Ҷȣ���ֵ
	Mat imgGray = Mat::zeros(imgHeight, imgWeight, CV_8UC1);
	Mat imgBin = Mat::zeros(imgHeight, imgWeight, CV_8UC1);

	//�ҶȻ�
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.114 * (float)img.at<Vec3b>(y, x)[2]
				+ 0.587 * (float)img.at<Vec3b>(y, x)[1]
				+ 0.299 * (float)img.at<Vec3b>(y, x)[0];
		}
	}


	const int grayScale = 256;//�Ҷ�ֵ

	int pixelCount[grayScale] = { 0 };//�Ҷ�ֱ��ͼ
	float pixelPro[grayScale] = { 0 };//�����Ҷ�ֵռ����ı���

	double w0, w1;//����/ǰ������ռ��
	double u0, u1;//ǰ��/����ƽ���Ҷ�ֵ
	double p0, p1;
	double g = 0;//��䷽��

	double max_g = 0;//�����䷽��
	double good_k = 0;//������ֵ
	int pixelSum = imgHeight * imgWeight;//������ֵ

	//ͳ��ͼƬ�и����Ҷ�ֵ�ĸ���
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			int val = imgGray.at<uchar>(y, x);
			pixelCount[val]++;
		}
	}

	//ͳ��ͼƬ�и����Ҷ�ֵ��ռ�ı���
	for (int i = 0; i < grayScale; ++i)
	{
		pixelPro[i] = 1.0 * pixelCount[i] / pixelSum;
	}

	//k:�ݶ���ֵ(0-255)
	for (int k = 0; k < grayScale; ++k)
	{
		w0 = w1 = u0 = u1 = g = 0;
		p0 = p1 = 0;
		//ǰ������������ [0-k][k+1-255]
		for (int i = 0; i < grayScale; ++i)
		{
			//�����ǰ����ֵС����ֵk�����ڱ�������֮����ǰ��
			if (i <= k)
			{
				//���㱳������ռ��
				w0 += pixelPro[i];
				//���㵱ǰ�Ҷ�ֵ�����ĸ���:�Ҷ�ֵ*�Ҷ�ֵ�����ĸ���
				p0 += (i * pixelPro[i]);

			}
			else
			{
				//���㱳������ռ��
				w1 += pixelPro[i];
				p1 += (i * pixelPro[i]);
			}
		}
		//����ƽ���Ҷ�ֵ��p0/w0
		u0 = p0 / w0;
		u1 = p1 / w1;
		//�������ڷ���
		g = (float)(w0 * w1 * pow((u0 - u1), 2));
		if (g > max_g)
		{
			max_g = g;
			good_k = k;

		}
	}

	printf_s("good k;%f\n", good_k);

	//ȡ����õ�kֵ����kֵ��Ϊ��ֵ���ж�ֵ��
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			if (imgGray.at<uchar>(y, x) > good_k)
				imgBin.at<uchar>(y, x) = 255;
			else
				imgBin.at<uchar>(y, x) = 0;
		}
	}

	//�����㣺��ʴ->����

	//��ʴ
	Mat tempImg;//��Ϊ�Ա�ͼ
	Mat imgErode = imgBin.clone();
	int time = 3;//ִ�д���
	for (int t = 0; t < time; ++t)
	{
		tempImg = imgErode.clone();
		for (int y = 0; y < imgHeight; ++y)
		{
			for (int x = 0; x < imgWeight; ++x)
			{
				//���ڴ�����������(x,y)=0��(x, y-1)��(x-1, y)��(x+1, y)��(x, y+1)�в�����һ����Ϊ255����(x,y)=0��
				if ((x > 0) && (tempImg.at<uchar>(y, x - 1) == 0))
				{
					imgErode.at<uchar>(y, x) = 0;
					continue;
				}
				if ((y > 0) && (tempImg.at<uchar>(y - 1, x) == 0))
				{
					imgErode.at<uchar>(y, x) = 0;
					continue;
				}
				if ((x < imgWeight - 1) && (tempImg.at<uchar>(y, x + 1) == 0))
				{
					imgErode.at<uchar>(y, x) = 0;
					continue;
				}
				if ((y < imgHeight - 1) && (tempImg.at<uchar>(y + 1, x) == 0))
				{
					imgErode.at<uchar>(y, x) = 0;
					continue;
				}
			}
		}
	}

	//����
	Mat imgDilate = imgErode.clone();
	time = 3;
	for (int t = 0; t < time; ++t)
	{
		tempImg = imgDilate.clone();
		for (int y = 0; y < imgHeight; ++y)
		{
			for (int x = 0; x < imgWeight; ++x)
			{
				//���ڴ�����������(x,y)=0��(x, y-1)��(x-1, y)��(x+1, y)��(x, y+1)�в�����һ��Ϊ255����(x,y)=255��
				if ((x > 0) && (tempImg.at<uchar>(y, x - 1) == 255))
				{
					imgDilate.at<uchar>(y, x) = 255;
					continue;
				}
				if ((y > 0) && (tempImg.at<uchar>(y - 1, x) == 255))
				{
					imgDilate.at<uchar>(y, x) = 255;
					continue;
				}
				if ((x < imgWeight - 1) && (tempImg.at<uchar>(y, x + 1) == 255))
				{
					imgDilate.at<uchar>(y, x) = 255;
					continue;
				}
				if ((y < imgHeight - 1) && (tempImg.at<uchar>(y + 1, x) == 255))
				{
					imgDilate.at<uchar>(y, x) = 255;
					continue;
				}
			}
		}
	}

	//��ñ:��ֵ-������
	Mat imgOut = Mat::zeros(imgHeight, imgWeight, CV_8UC1);
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			imgOut.at<uchar>(y, x) = imgBin.at<uchar>(y, x) - imgDilate.at<uchar>(y, x);
		}
	}

	cv::imshow("imgOut", imgOut);
	cv::waitKey(0);
	cv::destroyAllWindows();
}