#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"
using namespace cv;


Mat RGB_HSV(Mat rgb)
{
	int imgHeight = rgb.rows;
	int imgWeight = rgb.cols;
	Mat imgHSV = Mat::zeros(imgHeight, imgWeight, CV_32FC3);
	float R, G, B;
	float H, S, V;
	float Cmax, Cmin, delta;

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			//��ȡRGB
			R = (float)rgb.at<Vec3b>(y, x)[2] / 255;
			G = (float)rgb.at<Vec3b>(y, x)[1] / 255;
			B = (float)rgb.at<Vec3b>(y, x)[0] / 255;

			Cmax = fmax(R, fmax(G, B));
			Cmin = fmin(R, fmin(G, B));
			delta = Cmax - Cmin;

			if (delta == 0)
				H = 0;
			else if (Cmin == B)
				H = 60 * (G - R) / delta + 60;
			else if (Cmin == R)
				H = 60 * (B - G) / delta + 180;
			else if (Cmin == G)
				H = 60 * (R - B) / delta + 300;

			//����S(0-1)
			if (Cmax == 0)
				S = 0;
			else
				S = delta;
			//S = delta/Cmax;Ч������

		//����V(0-1)
			V = Cmax;

			imgHSV.at<Vec3f>(y, x)[0] = H;
			imgHSV.at<Vec3f>(y, x)[1] = S;
			imgHSV.at<Vec3f>(y, x)[2] = V;
		}
	}
	return imgHSV;
}

Mat HSV_RGB(Mat hsv)
{
	int imgHeight = hsv.rows;
	int imgWeight = hsv.cols;
	Mat imgRGB = Mat::zeros(imgHeight, imgWeight, CV_8UC3);
	float R, G, B;
	float H, S, V;

	double C, X, _H;
	//HSV->RGB
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			H = hsv.at<Vec3f>(y, x)[0];
			S = hsv.at<Vec3f>(y, x)[1];
			V = hsv.at<Vec3f>(y, x)[2];

			C = S;
			_H = H / 60;
			X = C * (1 - abs(fmod((H / 60), 2) - 1));
			R = G = B = V - C;

			if (_H < 1)
			{
				R += C;
				G += X;
			}
			else if (_H < 2)
			{
				R += X;
				G += C;
			}
			else if (_H < 3)
			{
				G += C;
				B += X;
			}
			else if (_H < 4)
			{
				G += X;
				B += C;
			}
			else if (_H < 5)
			{
				R += X;
				B += C;
			}
			else if (_H < 6)
			{
				R += C;
				B += X;
			}

			imgRGB.at<cv::Vec3b>(y, x)[0] = (uchar)(B * 255);
			imgRGB.at<cv::Vec3b>(y, x)[1] = (uchar)(G * 255);
			imgRGB.at<cv::Vec3b>(y, x)[2] = (uchar)(R * 255);
		}
	}
	return imgRGB;
}

//��ɫ׷��
Mat ColorTracking(Mat hsv)
{
	int imgHeight = hsv.rows;
	int imgWeight = hsv.cols;
	Mat out = Mat::zeros(imgHeight, imgWeight, CV_8UC1);
	//��ɫ��Χ180 - 260
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			int h = hsv.at<Vec3f>(y, x)[0];
			if (h > 180 && h < 260)
				out.at<uchar>(y, x) = 255;
			else
				out.at<uchar>(y, x) = 0;
		}
	}
	return out;
}

void A70(Mat img)
{
	Mat HSV = RGB_HSV(img);
	Mat out = ColorTracking(HSV);
	Mat RGB = HSV_RGB(HSV);

	imshow("img", img);
	imshow("out", out);
	imshow("RGB-HSV", HSV);
	imshow("HSV-RGB", RGB);

	waitKey(0);
	destroyAllWindows();
}

