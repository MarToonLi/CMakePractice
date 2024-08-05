#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"
#include <math.h>
using namespace cv;

void A9(void)
{
	printf_s("��˹�˲�\n");
	Mat imgSrc = imread("C:\\Users\\Administrator\\Desktop\\OpencvTestImg\\img512.png");
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);

	/*��˹�˲���������������Χ�����ذ��ո�˹�ֲ���Ȩƽ������ƽ�����������ģ���ά��Ȩֵͨ������Ϊ����ˣ�kernel�������˲�����filter����*/
	//��ͼƬ��С�����˲�����������ʱ����Ҫ����Χ��0

	int sigma = 1.3;//����
	const int kernelSize = 3;//����˴�С
	int kernelRadius = floor(kernelSize / 2);//����˰뾶
	float kernel[kernelSize][kernelSize];//����˾���
	double kernelSum = 0;
	int x_ = 0, y_ = 0;//�뾶�ڵ�����
	double PI = atan(1) * 4;

	//��������Ȩֵ
	for (int y = 0; y < kernelSize; ++y)
	{
		for (int x = 0; x < kernelSize; ++x)
		{
			x_ = x - kernelRadius;
			y_ = y - kernelRadius;
			//��ά��˹�ֲ�����
			kernel[y][x] = 1 / (2 * PI * sigma * sigma) * exp(-(x_ * x_ + y_ * y_) / (2 * sigma * sigma));
			kernelSum += kernel[y][x];
		}
	}
	//------------��ӡ�����------------
	printf_s("kernel-sum:%f\n", kernelSum);
	printf_s("δ��һ���ľ����\n");
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			printf_s("%f ", kernel[i][j]);
		}
		printf_s("\n");
	}

	double sum = 0;
	//��һ������ˣ�ʹ������Ȩֵ֮��Ϊ1
	for (int y = 0; y < kernelSize; y++) {
		for (int x = 0; x < kernelSize; x++) {
			kernel[y][x] = kernel[y][x] / kernelSum;
			sum += kernel[y][x];
		}
	}

	//------------��ӡ�����------------
	printf_s("kernel-sum:%f\n", sum);
	printf_s("��һ���Ժ�ľ����\n");
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			printf_s("%f ", kernel[i][j]);
		}
		printf_s("\n");
	}

	Mat imgtemp = Mat::zeros(imgHeight + 2 * kernelRadius, imgWidth + 2 * kernelRadius, CV_8UC3);
	//����һ��ԭͼ��������ӱ߿�
	for (int y = 0; y < imgHeight; y++)
		for (int x = 0; x < imgWidth; x++)
			for (int c = 0; c < channel; c++)
				imgtemp.at<Vec3b>(y + kernelRadius, x + kernelRadius)[c] = imgSrc.at<Vec3b>(y, x)[c];

	//ʹ�þ���˶�Դͼ����и�˹�˲�
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				double val = 0;
				for (int dy = -kernelRadius; dy < kernelRadius + 1; dy++)
					for (int dx = -kernelRadius; dx < kernelRadius + 1; dx++)
						//��ֹԽ��
						if (((y + dy) >= 0) && ((x + dx) >= 0))
							val += (double)imgtemp.at<Vec3b>(y + dy, x + dx)[c] * kernel[dy + kernelRadius][dx + kernelRadius];
				imgOut.at<Vec3b>(y, x)[c] = val;
			}
		}
	}
	imshow("src", imgSrc);
	imshow("out", imgOut);
	waitKey(0);
	destroyAllWindows();
}

