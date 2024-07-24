#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_31_40.h"
#include <math.h>
#include <time.h>

using namespace cv;


void A34(Mat img)
{
	//��ʱ
	clock_t start, stop;
	double duration;

	const int height = 128, width = 128;
	//��������
	std::complex<double> coef[height][width];

	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();

	//�ҶȻ�
	start = clock();//��ʼ��ʱ
	Mat imgGray = Mat::zeros(imgHeight, imgWidth, CV_8UC1);

	//�ҶȻ�
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.2126 * (float)img.at<Vec3b>(y, x)[2] \
				+ 0.7152 * (float)img.at<Vec3b>(y, x)[1] \
				+ 0.0722 * (float)img.at<Vec3b>(y, x)[0];
		}
	}
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);

	//DFT:��ɢ����Ҷ�任
	std::complex<double> i;//����:a+bi
	double val;//����ֵ
	double theta;
	//�ڶ��ڶ�άͼƬ����Ҫ�������������ϵĸ���Ҷ�任
	for (int y = 0; y < imgHeight; ++y)
	{
		printf_s("����Ҷ�任��y:%d��\n", y);
		for (int x = 0; x < imgWidth; ++x)
		{
			i.real(0);
			i.imag(0);
			/*printf_s("y:%d,x:%d\n",y,x);*/
			for (int _y = 0; _y < imgHeight; ++_y)
			{
				for (int _x = 0; _x < imgWidth; ++_x)
				{
					//��ȡͼ���ֵ(�����ź�):x(n)
					val = (double)imgGray.at<uchar>(_y, _x);
					//��=(2��kn)/N
					theta = -2 * PI * ((double)x * (double)_x / (double)imgWidth + (double)y * (double)_y / (double)imgHeight);
					//�任������ݣ�X(k)=��(N-1)(n=0)[x(n)cos(��)+jsin(��)];
					i += std::complex<double>(cos(theta), sin(theta)) * val;
				}
			}
			i /= sqrt(imgHeight * imgWidth);
			coef[y][x] = i;
		}
	}

	printf_s("����Ҷ�任�������\n");

	//��ͨ�˲�
	int r = imgHeight / 2;
	double pass_r = 0.5;
	int filter_d = (int)((double)r * pass_r);
	for (int j = 0; j < imgHeight / 2; ++j)
	{
		for (int i = 0; i < imgWidth / 2; ++i)
		{
			if (sqrt(i * i + j * j) <= filter_d)
			{
				coef[j][i] = 0;
				coef[j][imgWidth - i] = 0;
				coef[imgHeight - j][i] = 0;
				coef[imgHeight - j][imgWidth - i] = 0;
			}
		}
	}

	double g;
	std::complex<double> G;

	//IDFT ��ɢ����Ҷ��任
	for (int y = 0; y < imgHeight; ++y)
	{
		printf_s("����Ҷ��任��y:%d��\n", y);
		for (int x = 0; x < imgWidth; ++x)
		{
			i.real(0);
			i.imag(0);

			for (int _y = 0; _y < imgHeight; ++_y)
			{
				for (int _x = 0; _x < imgWidth; ++_x)
				{
					//��ȡ����Ҷ�任���������
					G = coef[_y][_x];
					//��=(2��kn)/N
					theta = 2 * PI * ((double)_x * (double)x / (double)imgWidth + (double)_y * (double)y / (double)imgHeight);
					//�任������ݣ�X(k)=��(N-1)(n=0)[x(n)cos(��)+jsin(��)];
					i += std::complex<double>(cos(theta), sin(theta)) * G;
				}
			}
			g = std::abs(i) / sqrt(height * width);
			imgOut.at<uchar>(y, x) = (uchar)g;
		}
	}
	/* ������ʱ */
	stop = clock();
	duration = ((double)(stop - start)) / CLK_TCK;
	printf("����ʱ�䣺%f\n", duration);
	printf_s("����Ҷ��任�������\n");
	imshow("imgSrc", img);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}