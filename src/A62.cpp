#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_61_70.h"
#include <math.h>

void A62(Mat img)
{
	int H = img.rows;
	int W = img.cols;
	int channel = img.channels();
	int val = 0;
	//�Ҷȶ�ֵ��
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

	const int grayScale = 256;//�Ҷ�ֵ

	int pixelCount[grayScale] = { 0 };//�Ҷ�ֱ��ͼ
	float pixelPro[grayScale] = { 0 };//�����Ҷ�ֵռ����ı���

	double w0, w1;//����/ǰ������ռ��
	double u0, u1;//ǰ��/����ƽ���Ҷ�ֵ
	double p0, p1;
	double g = 0;//��䷽��

	double max_g = 0;//�����䷽��
	double good_k = 0;//������ֵ
	int pixelSum = H * W;//������ֵ

	//ͳ��ͼƬ�и����Ҷ�ֵ�ĸ���
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
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
	Mat img8 = Mat::zeros(H, W, CV_8UC1);
	//ȡ����õ�kֵ����kֵ��Ϊ��ֵ���ж�ֵ��
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			//��������з�ת0-255
			if (imgGray.at<uchar>(y, x) > good_k)
				img8.at<uchar>(y, x) = 0;
			else
				img8.at<uchar>(y, x) = 1;

			if (imgGray.at<uchar>(y, x) > good_k)
				imgBin.at<uchar>(y, x) = 1;
			else
				imgBin.at<uchar>(y, x) = 0;
		}
	}

	Mat imgOut = Mat::zeros(H, W, CV_8UC3);
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			//printf_s("y:%d x:%d\n",y,x);
			//����Ǻ�ɫ������
			if (imgBin.at<uchar>(y, x) == 0)
				continue;
			int s = 0;
			//����
			s += (img8.at<uchar>(y, MIN(x + 1, W - 1)) - img8.at<uchar>(y, MIN(x + 1, W - 1)) * img8.at<uchar>(MAX(y - 1, 0), MIN(x + 1, W - 1)) * img8.at<uchar>(MAX(y - 1, 0), x));
			s += (img8.at<uchar>(MAX(y - 1, 0), x) - img8.at<uchar>(MAX(y - 1, 0), x) * img8.at<uchar>(MAX(y - 1, 0), MAX(x - 1, 0)) * img8.at<uchar>(y, MAX(x - 1, 0)));
			s += (img8.at<uchar>(y, MAX(x - 1, 0)) - img8.at<uchar>(y, MAX(x - 1, 0)) * img8.at<uchar>(MIN(y + 1, H - 1), MAX(x - 1, 0)) * img8.at<uchar>(MIN(y + 1, H - 1), x));
			s += (img8.at<uchar>(MIN(y + 1, H - 1), x) - img8.at<uchar>(MIN(y + 1, H - 1), x) * img8.at<uchar>(MIN(y + 1, H - 1), MIN(x + 1, W - 1)) * img8.at<uchar>(y, MIN(x + 1, W - 1)));

			if (s == 0)
			{
				imgOut.at<Vec3b>(y, x)[0] = 0;
				imgOut.at<Vec3b>(y, x)[1] = 0;
				imgOut.at<Vec3b>(y, x)[2] = 255;
			}
			else if (s == 1)
			{
				imgOut.at<Vec3b>(y, x)[0] = 0;
				imgOut.at<Vec3b>(y, x)[1] = 255;
				imgOut.at<Vec3b>(y, x)[2] = 0;
			}
			else if (s == 2)
			{
				imgOut.at<Vec3b>(y, x)[0] = 255;
				imgOut.at<Vec3b>(y, x)[1] = 0;
				imgOut.at<Vec3b>(y, x)[2] = 0;
			}
			else if (s == 3)
			{
				imgOut.at<Vec3b>(y, x)[0] = 255;
				imgOut.at<Vec3b>(y, x)[1] = 255;
				imgOut.at<Vec3b>(y, x)[2] = 0;
			}
			else if (s == 4)
			{
				imgOut.at<Vec3b>(y, x)[0] = 255;
				imgOut.at<Vec3b>(y, x)[1] = 0;
				imgOut.at<Vec3b>(y, x)[2] = 255;
			}
		}
	}
	//imwrite("imgout.jpg", imgOut);
	cv::imshow("img", img);
	cv::imshow("imgBin", imgBin);
	cv::imshow("imgOut", imgOut);
	cv::waitKey(0);
	cv::destroyAllWindows();
}