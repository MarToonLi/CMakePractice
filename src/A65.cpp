#include <iostream>
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

	//��ת��ɫ��0��Ϊ�ߣ���1��Ϊ����
	Mat imgRe = Mat::zeros(H, W, CV_8UC1);

	//ȡ����õ�kֵ����kֵ��Ϊ��ֵ���ж�ֵ��
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
				//�ų���ɫ
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
				//�Ұ�Ȧ
				if (imgRe.at<uchar>(y - 1, x) + imgRe.at<uchar>(y, x + 1) + imgRe.at<uchar>(y + 1, x) < 1)
					continue;
				// ���Ȧ
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
				//�ų���ɫ
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

				//�ϰ�Ȧ
				if (imgRe.at<uchar>(y - 1, x) + imgRe.at<uchar>(y, x + 1) + imgRe.at<uchar>(y, x - 1) < 1)
					continue;

				// �°�Ȧ
				if (imgRe.at<uchar>(y - 1, x) + imgRe.at<uchar>(y + 1, x) + imgRe.at<uchar>(y, x - 1) < 1)
					continue;

				imgBin.at<uchar>(y, x) = 0;
				imgRe.at<uchar>(y, x) = 1;
				find = true;
			}
		}
	}
	//��������
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