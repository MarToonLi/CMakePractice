#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_21_30.h"

using namespace cv;

void A26(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();

	//���ű���
	double rx;
	double ry;
	rx = ry =2;

	//���ź�Ŀ��
	int resizeHeight = (int)(imgHeight * ry);
	int resizeWidth = (int)(imgWidth * rx);
	Mat imgOut = Mat::zeros(resizeHeight, resizeWidth, CV_8UC3);

	printf_s("h:%d,w:%d\n", imgHeight, imgWidth);
	printf_s("rh:%d,rw:%d\n", resizeHeight, resizeWidth);

	//����ֵ
	int xBefore, yBefore;
	double dy, dx;
	double val;
	for (int y = 0; y < resizeHeight; ++y)
	{
		yBefore = (int)floor(y / ry);
		yBefore = fmin(yBefore, imgHeight - 1);
		dy = y / ry - yBefore;

		for (int x = 0; x < resizeWidth; ++x)
		{
			xBefore = (int)floor(x / rx);
			xBefore = fmin(xBefore, imgWidth - 1);
			dx = x / rx - xBefore;

			for (int c = 0; c < channel; ++c)
			{
				val = (1. - dx) * (1. - dy) * imgSrc.at<Vec3b>(yBefore, xBefore)[c] +
					dx * (1. - dy) * imgSrc.at<Vec3b>(yBefore, MIN(xBefore + 1, imgWidth - 1))[c] +
					(1. - dx) * dy * imgSrc.at<Vec3b>(MIN(yBefore + 1, imgHeight - 1), xBefore)[c] +
					dx * dy * imgSrc.at<Vec3b>(MIN(yBefore + 1, imgHeight - 1), xBefore)[c];
				imgOut.at<Vec3b>(y, x)[c] = (uchar)val;
			}
		}
	}

	imshow("imgSrc", img);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}