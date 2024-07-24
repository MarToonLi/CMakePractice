#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

float _Clip_(float value, float min, float max)
{
	return fmin(fmax(value, 0), 255);
}

Mat _ImgGray_(Mat img)
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
	return imgGray;
}

//��˹�˲�
Mat _GaussianFilter_(Mat img, double Sigma)
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
				imgGaussianOut.at<uchar>(y, x) = (uchar)_Clip_(val, 0, 255);
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
					imgGaussianOut.at<Vec3b>(y, x) = (uchar)_Clip_(val, 0, 255);
				}

			}
		}
	}
	return imgGaussianOut;
}

//Sobel filter
//��ֱ
Mat _SobelFilterV_(Mat img)
{
	int imgHeight = img.rows;
	int imgWidth = img.cols;

	cv::Mat out = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	
	const int kSize = 3;
	int kRadius = floor(kSize / 2);//����뾶
	// prepare kernel:��ֱ
	double kernel[kSize][kSize] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

	double v = 0;

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			double valV = 0;
			for (int dy = -kRadius; dy < kRadius + 1; dy++)
			{
				for (int dx = -kRadius; dx < kRadius + 1; dx++)
				{
					if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
					{
						valV += (double)img.at<uchar>(y + dy, x + dx) * kernel[kRadius + dy][kRadius + dx];
					}
				}
			}
			out.at<uchar>(y, x) = (uchar)_Clip_(valV, 0, 255);
		}
	}
	return out;


}
//ˮƽ
Mat _SobelFilterH_(Mat img)
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
			for (int dy = -kRadius; dy < kRadius + 1; dy++)
			{
				for (int dx = -kRadius; dx < kRadius + 1; dx++)
				{
					if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
					{
						valH += (double)img.at<uchar>(y + dy, x + dx) * kSobelHorizontal[kRadius + dy][kRadius + dx];
					}
				}
			}
			imgOutH.at<uchar>(y, x) = (uchar)_Clip_(valH, 0, 255);
		}
	}
	return imgOutH;
}

//���������ϵ��˲�ͼƬ:��
Mat _GetEdge_(Mat imgX, Mat imgY)
{
	int imgHeight = imgX.rows;
	int imgWidth = imgX.cols;
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	double fx, fy;
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			fx = (double)imgX.at<uchar>(y, x);
			fy = (double)imgY.at<uchar>(y, x);
			imgOut.at<uchar>(y, x) = (uchar)_Clip_(sqrt(fx * fx + fy * fy), 0, 255);
		}
	}
	return imgOut;
}

//���������ϵ��˲�ͼƬ����

Mat _GetAngel_(Mat imgX, Mat imgY)
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
			angle = angle / PI * 180;

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
Mat _NMS_(Mat angle, Mat edge)
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

Mat Histerisis(Mat edge, int HT, int LT)
{
	int imgHeight = edge.rows;
	int imgWidth = edge.cols;
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);

	uchar nowPixel = 0;

	for (int y = 0; y < imgHeight; y++)
	{
		for (int x = 0; x < imgWidth; x++)
		{
			//printf_s("y:%d,x:%d\n", y, x);
			nowPixel = edge.at<uchar>(y, x);
			if (nowPixel >= HT)
			{
				imgOut.at<uchar>(y, x) = 255;
			}

			else if (nowPixel > LT)
			{
				for (int dy = -1; dy < 2; dy++)
				{
					for (int dx = -1; dx < 2; dx++)
					{
						if (edge.at<uchar>(fmin(fmax(y + dy, 0), 255), fmin(fmax(x + dx, 0), 255)) >= HT)
						{
							imgOut.at<uchar>(y, x) = 255;
						}
					}
				}
			}
		}
	}
	return imgOut;
}
void A43(Mat img)
{
	//ͼƬ�ҶȻ�
	Mat imgGray = _ImgGray_(img);

	//��˹�˲�
	Mat imgGaussian = _GaussianFilter_(imgGray, 1.4);

	//sobel filter:��������
	Mat imgx = _SobelFilterH_(imgGaussian);
	Mat imgy = _SobelFilterV_(imgGaussian);
	//�ߣ���
	Mat edge = _GetEdge_(imgx, imgy);
	Mat angel = _GetAngel_(imgx, imgy);

	//�Ǽ���ֵ����
	edge = _NMS_(angel, edge);
	//���ӱ�
	edge = Histerisis(edge, 80, 20);

	imshow("imgSrc", img);
	imshow("imgGray", imgGray);

	imshow("imgGaussian", imgGaussian);
	imshow("imgx", imgx);
	imshow("imgy", imgy);
	imshow("edge", edge);
	imshow("angel", angel);
	waitKey(0);
	destroyAllWindows();
}