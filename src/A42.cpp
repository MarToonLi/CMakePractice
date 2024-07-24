#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

double _PI = atan(1) * 4;


float _Clip(float value, float min, float max)
{
	return fmin(fmax(value, 0), 255);
}

//ͼ��ҶȻ�	
Mat _ImgGray(Mat img)
{
	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();
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
	imshow("imgGray", imgGray);
	return imgGray;
}
//��˹�˲�
Mat _GaussianFilter(Mat img, double Sigma)
{
	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();
	//��ͨ��������ͨ��
	Mat imgGaussianOut;
	if (channel == 1)
		imgGaussianOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	else
		imgGaussianOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);
	const int kSize = 5;

	int kRadius = floor((double)kSize / 2);//����뾶
	double kernel[kSize][kSize];
	double kernel_sum = 0;
	double _x, _y;

	//��˹�˲�,��������
	for (int y = 0; y < kSize; y++)
	{
		for (int x = 0; x < kSize; x++)
		{
			_y = y - kRadius;
			_x = x - kRadius;
			kernel[y][x] = 1 / (2 * PI * Sigma * Sigma) * exp(-(_x * _x + _y * _y) / (2 * Sigma * Sigma));
			kernel_sum += kernel[y][x];
		}
	}

	for (int y = 0; y < kSize; y++)
	{
		for (int x = 0; x < kSize; x++)
		{
			kernel[y][x] /= kernel_sum;
		}
	}

	double val = 0;
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			//����ǵ�ͨ��
			if (channel == 1)
			{
				val = 0;
				for (int dy = -kRadius; dy < kRadius + 1; dy++)
				{
					for (int dx = -kRadius; dx < kRadius + 1; dx++)
					{
						if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
						{
							val += img.at<uchar>(y + dy, x + dx) * kernel[kRadius + dy][kRadius + dx];
						}

					}
				}
				imgGaussianOut.at<uchar>(y, x) = (uchar)_Clip(val, 0, 255);
			}
			//�������ͨ��
			else
			{
				for (int c = 0; c < channel; ++c)
				{
					val = 0;
					for (int dy = -kRadius; dy < kRadius + 1; dy++)
					{
						for (int dx = -kRadius; dx < kRadius + 1; dx++)
						{
							if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
							{
								val += img.at<Vec3b>(y + dy, x + dx)[c] * kernel[kRadius + dy][kRadius + dx];
							}

						}
					}
					imgGaussianOut.at<Vec3b>(y, x) = (uchar)_Clip(val, 0, 255);
				}

			}
		}
	}

	imshow("imgGaussianOut", imgGaussianOut);
	return imgGaussianOut;

}

//Sobel filter
//��ֱ
Mat _SobelFilterV(Mat img)
{
	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();
	Mat imgOutV = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	const int kSize = 3;
	//����뾶
	int kRadius = floor((double)kSize / 2);//����뾶
	//�����
	//���þ����:��ֱ
	double kSobelVertical[kSize][kSize] = { {1, 2, 1},
											{0, 0, 0},
											{-1, -2, -1} };

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			double valH = 0;
			double valV = 0;
			for (int dy = -kRadius; dy < kRadius + 1; dy++)
			{
				for (int dx = -kRadius; dx < kRadius + 1; dx++)
				{
					if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
					{
						valV += img.at<uchar>(y + dy, x + dx) * kSobelVertical[kRadius + dy][kRadius + dx];
					}

				}
			}
			imgOutV.at<uchar>(y, x) = (uchar)_Clip(valV, 0, 255);
		}
	}
	imshow("imgOutV", imgOutV);
	return imgOutV;

}
//ˮƽ
Mat _SobelFilterH(Mat img)
{
	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();

	Mat imgOutH = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	const int kSize = 3;
	//����뾶
	int kRadius = floor((double)kSize / 2);//����뾶
	//�����
	//���þ����:ˮƽ
	double kSobelHorizontal[kSize][kSize] = { {1, 0, 1},
											{2, 0, -2},
											{-1, 0, -1} };


	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			double valH = 0;
			double valV = 0;
			for (int dy = -kRadius; dy < kRadius + 1; dy++)
			{
				for (int dx = -kRadius; dx < kRadius + 1; dx++)
				{
					if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
					{
						valH += img.at<uchar>(y + dy, x + dx) * kSobelHorizontal[kRadius + dy][kRadius + dx];
					}
				}
			}
			imgOutH.at<uchar>(y, x) = (uchar)_Clip(valH, 0, 255);
		}
	}
	imshow("imgOutH", imgOutH);
	return imgOutH;
}

//���������ϵ��˲�ͼƬ:��
Mat _GetEdge(Mat imgX, Mat imgY)
{
	int imgHeight = imgX.rows;
	int imgWidth = imgX.cols;
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	double fx, fy;
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			fx = imgX.at<uchar>(y, x);
			fy = imgY.at<uchar>(y, x);
			imgOut.at<uchar>(y, x) = (uchar)_Clip(sqrt(fx * fx + fy * fy), 0, 255);
		}
	}
	return imgOut;
}

//���������ϵ��˲�ͼƬ����

Mat _GetAngel(Mat imgX, Mat imgY)
{
	int imgHeight = imgX.rows;
	int imgWidth = imgX.cols;

	int channel = imgX.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	double fx, fy;
	double angle;
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			fx = fmax((double)imgX.at<uchar>(y, x), 0.000001);
			fy = (double)imgY.at<uchar>(y, x);

			angle = atan2(fy, fx);
			angle = angle / _PI * 180;

			if (angle < -22.5)
			{
				angle = 180 + angle;
			}
			else if (angle >= 157.5)
			{
				angle = angle - 180;
			}

			if (angle <= 22.5)
			{
				imgOut.at<uchar>(y, x) = 0;
			}
			else if (angle <= 67.5)
			{
				imgOut.at<uchar>(y, x) = 45;
			}
			else if (angle <= 112.5)
			{
				imgOut.at<uchar>(y, x) = 90;
			}
			else
			{
				imgOut.at<uchar>(y, x) = 135;
			}
		}
	}
	return imgOut;
}

//�Ǽ���ֵ���ƣ�NMS
Mat NMS(Mat angle, Mat edge)
{
	int imgHeight = angle.rows;
	int imgWeidth = angle.cols;
	int channle = angle.channels();

	int dx1, dx2, dy1, dy2;
	int nowAngle;
	Mat imgOut = Mat::zeros(imgHeight, imgWeidth, CV_8UC1);
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeidth; ++x)
		{
			nowAngle = angle.at<uchar>(y, x);
			// angle condition
			if (nowAngle == 0) {
				dx1 = -1;
				dy1 = 0;
				dx2 = 1;
				dy2 = 0;
			}
			else if (nowAngle == 45) {
				dx1 = -1;
				dy1 = 1;
				dx2 = 1;
				dy2 = -1;
			}
			else if (nowAngle == 90) {
				dx1 = 0;
				dy1 = -1;
				dx2 = 0;
				dy2 = 1;
			}
			else {
				dx1 = -1;
				dy1 = -1;
				dx2 = 1;
				dy2 = 1;
			}

			if (x == 0) {
				dx1 = fmax(dx1, 0);
				dx2 = fmax(dx2, 0);
			}
			if (x == (imgWeidth - 1)) {
				dx1 = fmin(dx1, 0);
				dx2 = fmin(dx2, 0);
			}
			if (y == 0) {
				dy1 = fmax(dy1, 0);
				dy2 = fmax(dy2, 0);
			}
			if (y == (imgHeight - 1)) {
				dy1 = fmin(dy1, 0);
				dy2 = fmin(dy2, 0);
			}

			// if pixel is max among adjuscent pixels, pixel is kept
			if (fmax(fmax(edge.at<uchar>(y, x), edge.at<uchar>(y + dy1, x + dx1)), edge.at<uchar>(y + dy2, x + dx2)) == edge.at<uchar>(y, x)) {
				imgOut.at<uchar>(y, x) = edge.at<uchar>(y, x);
			}
		}
	}
	return imgOut;
}

void A42(Mat img)
{
	Mat out;
	//ͼƬ�ҶȻ�
	out = _ImgGray(img);
	//��˹�˲�
	out = _GaussianFilter(out, 1.4);
	//sobel filter:��������
	Mat imgx = _SobelFilterH(out);
	Mat imgy = _SobelFilterV(out);
	//�ߣ���
	Mat edge = _GetEdge(imgx, imgy);
	Mat angel = _GetAngel(imgx, imgy);
	//�Ǽ���ֵ����
	edge = NMS(edge, angel);

	imshow("imgSrc", img);
	imshow("edge", edge);
	imshow("angel", angel);
	waitKey(0);
	destroyAllWindows();
}