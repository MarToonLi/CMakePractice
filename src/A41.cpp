#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

float Clip(float value, float min, float max)
{
	return fmin(fmax(value, 0), 255);
}

//ͼ��ҶȻ�	 
Mat ImgGray(Mat img)
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
	//imshow("imgGray", imgGray);
	return imgGray;
}
//��˹�˲�
Mat GaussianFilter(Mat img, double Sigma)
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
				imgGaussianOut.at<uchar>(y, x) = (uchar)Clip(val, 0, 255);
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
					imgGaussianOut.at<Vec3b>(y, x) = (uchar)Clip(val, 0, 255);
				}

			}
		}
	}

	//imshow("imgGaussianOut", imgGaussianOut);
	return imgGaussianOut;

}


//Sobel filter
//��ֱ
Mat SobelFilterV(Mat img)
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
			imgOutV.at<uchar>(y, x) = (uchar)Clip(valV, 0, 255);
		}
	}
	//imshow("imgOutV", imgOutV);
	return imgOutV;

}
//ˮƽ
Mat SobelFilterH(Mat img)
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
						valH += img.at<uchar>(y + dy, x + dx) * kSobelHorizontal[kRadius + dy][kRadius + dx];
					}
				}
			}
			imgOutH.at<uchar>(y, x) = (uchar)Clip(valH, 0, 255);
		}
	}
	//imshow("imgOutH", imgOutH);
	return imgOutH;
}

//���������ϵ��˲�ͼƬ:��
Mat GetEdge(Mat imgX, Mat imgY)
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
			imgOut.at<uchar>(y, x) = (uchar)Clip(sqrt(fx * fx + fy * fy), 0, 255);
		}
	}
	return imgOut;
}

//���������ϵ��˲�ͼƬ����

Mat GetAngel(Mat imgX, Mat imgY)
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
void A41(Mat img)
{
	Mat out;
	//ͼƬ�ҶȻ�
	out = ImgGray(img);
	//��˹�˲�
	out = GaussianFilter(out,1.4);
	//sobel filter:��������
	Mat imgx = SobelFilterH(out);
	Mat imgy = SobelFilterV(out);
	//�ߣ���
	Mat edge = GetEdge(imgx,imgy);
	Mat angel = GetAngel(imgx,imgy);

	imshow("imgSrc", img);
	imshow("edge", edge);
	imshow("angel", angel);
	waitKey(0);
	destroyAllWindows();
}