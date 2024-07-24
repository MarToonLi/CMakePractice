#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_51_60.h"
#include <math.h>
#include <time.h>
#include <vector>
#include <map>

using namespace cv;

void A59(Mat img)
{
	int imgHeight = img.rows;
	int imgWeight = img.cols;
	int channel = img.channels();
	int val = 0;
	//�Ҷȶ�ֵ��
	Mat imgGray = Mat::zeros(imgHeight, imgWeight, CV_8UC1);
	Mat imgBin = Mat::zeros(imgHeight, imgWeight, CV_8UC1);

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

	int label = 0;
	uchar up = 0, left = 0, leftup = 0, rightup = 0;

	int** labelSet;

	//���ٿռ�:��=y
	labelSet = new int* [imgHeight];

	//���ٿռ䣺��=x
	for (int i = 0; i < imgHeight; ++i)
		labelSet[i] = new int[imgWeight];

	for (int i = 0; i < imgHeight; i++)
		for (int j = 0; j < imgWeight; j++)
			labelSet[i][j] = 0;

	//bgr
	Scalar white(255, 255, 255);

	Scalar blue(255, 0, 0);
	Scalar greed(0, 255, 0);
	Scalar red(0, 0, 255);
	Scalar yellow(0, 255, 255);
	Scalar pink(128, 0, 255);
	Scalar purple(255, 0, 128);
	Scalar origin(0, 128, 255);

	Scalar colorSet[] = { white, blue ,greed ,red ,yellow,pink,purple ,origin };
	Scalar drawColor;
	int k = 0;
	Mat temp = img.clone();
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			val = (int)imgBin.at<uchar>(y, x);
			//����ǰ�ɫ��255
			if (val == 255)
			{
				Point p(x, y);
				circle(temp, p, 0, red);
				cv::imshow("temp", temp);
				cv::waitKey(5);

				if (y >= 1 && x >= 1)
				{
					//���½�
					if (x == imgWeight - 1 && y == imgHeight - 1)
					{
						up = (int)imgBin.at<uchar>(y - 1, x);
						left = (int)imgBin.at<uchar>(y, x - 1);
						leftup = (int)imgBin.at<uchar>(y - 1, x - 1);
						rightup = 0;
					}
					up = (int)imgBin.at<uchar>(y - 1, x);
					leftup = (int)imgBin.at<uchar>(y - 1, x - 1);
					left = (int)imgBin.at<uchar>(y, x - 1);
					rightup = (int)imgBin.at<uchar>(y - 1, x + 1);
				}
				//��һ��
				else if (x == 0 && y >= 1)
				{
					up = (int)imgBin.at<uchar>(y - 1, x);
					leftup = 0;
					left = 0;
					rightup = (int)imgBin.at<uchar>(y, x + 1);
				}
				//��һ��
				else if (x >= 1 && y == 0)
				{
					up = 0;
					leftup = 0;
					left = (int)imgBin.at<uchar>(y, x - 1);
					rightup = 0;
				}
				//���Ͻ�
				else
				{
					up = 0;
					leftup = 0;
					left = 0;
					rightup = 0;
				}
				printf_s("y:%d x:%d left:%d leftup:%d up:%d rightup:%d\n", y, x, left, leftup, up, rightup);
				//����������ֵΪ0�������lable
				if (up == 0 && left == 0 && leftup == 0 && rightup == 0)
				{
					label++;
					imgBin.at<uchar>(y, x) = label;
					printf_s("label:%d\n", label);
					drawColor = colorSet[label];

					//����
					Point p(x, y);
					circle(temp, p, 0, drawColor);
					cv::imshow("temp", temp);
					cv::waitKey(5);
					continue;
				}
				if (left > 0 && rightup > 0)
				{
					int min = MIN(rightup, left);
					if (rightup == 0)
						min = left;
					if (left == 0)
						min = rightup;
					left = rightup = min;
					imgBin.at<uchar>(y, x) = min;

					drawColor = colorSet[min];
					Point p(x, y);
					circle(temp, p, 0, drawColor);
					cv::imshow("temp", temp);
					cv::waitKey(5);
				}
				if (leftup > 0 && rightup > 0)
				{
					int min = MIN(rightup, leftup);
					if (rightup == 0)
						min = leftup;
					if (leftup == 0)
						min = rightup;
					leftup = rightup = min;
					imgBin.at<uchar>(y, x) = min;

					drawColor = colorSet[min];
					Point p(x, y);
					circle(temp, p, 0, drawColor);
					cv::imshow("temp", temp);
					cv::waitKey(5);
				}
				//�������һ����Ϊ0(��)��ѡ����С��labelΪ������label
				else
				{
					int min = MIN(MIN(MIN(up, left), leftup), rightup);

					if (rightup == 0)
						min =left;
					imgBin.at<uchar>(y, x) = min;

					drawColor = colorSet[min];
					Point p(x, y);
					circle(temp, p, 0, drawColor);
					cv::imshow("temp", temp);
					cv::waitKey(5);
				}
			}
			//����Ƿǰ�ɫ�������һ������
			else
				continue;

		}
	}
	printf_s("ɨ�����,label:%d\n", label);

	cv::imshow("img", img);
	cv::imshow("imgBin", imgBin);
	cv::waitKey(0);
	cv::destroyAllWindows();
}