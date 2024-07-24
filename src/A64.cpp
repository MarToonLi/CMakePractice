#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_61_70.h"
#include <math.h>

using namespace cv;

int Calculate8Connections(Mat img8, int y, int x, int H, int W)
{
	int s = 0;
	s += (img8.at<uchar>(y, MIN(x + 1, W - 1)) - img8.at<uchar>(y, MIN(x + 1, W - 1)) * img8.at<uchar>(MAX(y - 1, 0), MIN(x + 1, W - 1)) * img8.at<uchar>(MAX(y - 1, 0), x));
	s += (img8.at<uchar>(MAX(y - 1, 0), x) - img8.at<uchar>(MAX(y - 1, 0), x) * img8.at<uchar>(MAX(y - 1, 0), MAX(x - 1, 0)) * img8.at<uchar>(y, MAX(x - 1, 0)));
	s += (img8.at<uchar>(y, MAX(x - 1, 0)) - img8.at<uchar>(y, MAX(x - 1, 0)) * img8.at<uchar>(MIN(y + 1, H - 1), MAX(x - 1, 0)) * img8.at<uchar>(MIN(y + 1, H - 1), x));
	s += (img8.at<uchar>(MIN(y + 1, H - 1), x) - img8.at<uchar>(MIN(y + 1, H - 1), x) * img8.at<uchar>(MIN(y + 1, H - 1), MIN(x + 1, W - 1)) * img8.at<uchar>(y, MIN(x + 1, W - 1)));
	return s;
}

void A64(Mat img)
{
	int H = img.rows;
	int W = img.cols;
	int channel = img.channels();
	int val = 0;
	//�Ҷȶ�ֵ����0-1��
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
			//8����������������з�ת0-255
			if (imgGray.at<uchar>(y, x) > good_k)
			{
				imgBin.at<uchar>(y, x) = 1;
				img8.at<uchar>(y, x) = 0;
			}
			else
			{
				imgBin.at<uchar>(y, x) = 0;
				img8.at<uchar>(y, x) = 1;
			}

		}
	}
	//��ɾ������Ӧ������6������
	//1.p�ĻҶ�ֵΪ255����p���Ǳ�����
	//	2.x0��x2��x4��x6��ȫΪ255������Ȼɾ��p�������ͼ����ģ�
	//	3.x0 - x7������2��255������ֻ��һ��255����pΪͼ���ĳ���˵㣬����ɾ������Ӱ��ͼ�����ʣ���û��255����pΪ�����㣬Ҳ����ɾ����
	//	4.p��8��ͨ����������Ϊ1����Nc��p�� = 1����ʱp��Ϊ��ɾ���㣻
	//	5.����x2�����ɾ����x2 = 0ʱ��p����ͨ������Ϊ1��
	//	6.����x4�����ɾ����x4 = 0ʱ��p����ͨ������Ϊ1��

	Mat out = imgBin.clone();
	Mat tmp;
	bool flag = true;

	while (flag)
	{
		flag = false;
		tmp = out.clone();
		for (int y = 0; y < H; ++y)
		{
			for (int x = 0; x < W; ++x)
			{
				if (out.at<uchar>(y, x) == 0)
					img8.at<uchar>(y, x) = 1;
				else
					img8.at<uchar>(y, x) = 0;
			}
		}

		for (int y = 0; y < H; ++y)
		{
			for (int x = 0; x < W; ++x)
			{
				//��ɫ������
				if (tmp.at<uchar>(y, x) == 0)
					continue;

				//4������������һ��0
				if (tmp.at<uchar>(MAX(y - 1, 0), x) * tmp.at<uchar>(MIN(y + 1, H - 1), x)* tmp.at<uchar>(y, MAX(x - 1, 0) * tmp.at<uchar>(y, MIN(x + 1, W - 1)) !=0))
					continue;
				
				//x1-x8 �У�������2��Ϊ1
				int sum = 0;
				for (int _y = -1; _y < 2; ++_y)
				{
					for (int _x = -1; _x < 2; ++_x)
					{
						if (_y == 0 && _x == 0)
							continue;
						sum += tmp.at<uchar>(y + _y, x + _x);
					}
				}
				if (sum <= 1)
					continue;

				//p��8��ͨ������Ϊ1��
				//���㣨y,x����8������
				int s = Calculate8Connections(img8, y, x, H, W);
				if (s != 1)
					continue;

				// ����5: ����p2�ѱ��ɾ����1��������p2Ϊ������1����p��������=1 
				//��img8�� 0��ǰ����1�Ǳ���
				img8.at<uchar>(y - 1, x) = 1;
				s = Calculate8Connections(img8, y, x, H, W);
				img8.at<uchar>(y - 1, x) = 0;
				if (s != 1)
					continue;
				
				// ����6: ����p4�ѱ��ɾ��������p4Ϊ������p��������=1 
				img8.at<uchar>(y, x - 1) = 1;
				s = Calculate8Connections(img8, y, x, H, W);
				img8.at<uchar>(y, x - 1) = 0;

				if (s != 1)
					continue;

				printf_s("ִ��\n");
				out.at<uchar>(y, x) = 0;//��������������ɾ�����
				flag = true;
			}
		}
	}

	//��������
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			if (out.at<uchar>(y, x) == 0)
				continue;
			out.at<uchar>(y, x) = out.at<uchar>(y, x) * 255;
		}
	}
	cv::imshow("img", img);
	cv::imshow("out", out);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

