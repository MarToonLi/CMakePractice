#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_21_30.h"

using namespace cv;

void A25(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	double val;
	//���ű���
	double rx = 1.5;
	double ry = 1.5;
	//���ź�Ŀ��
	int resizeHeight = (int)(imgHeight * ry);
	int resizeWidth = (int)(imgWidth * rx);

	printf_s("h:%d,w:%d\n",imgHeight,imgWidth);
	printf_s("rh:%d,rw:%d\n",resizeHeight,resizeWidth);
	Mat imgOut = Mat::zeros(resizeHeight, resizeWidth, CV_8UC3);
	//����ֵ
	int xBefore;
	int yBefore;

	//round(x)����x��������������ֵ��

	for (int y = 0; y < resizeHeight; ++y)
	{
		yBefore = (int)round(y / ry);//���y�����ű���
		yBefore = fmin(yBefore, imgHeight - 1);

		for (int x = 0; x < resizeWidth; ++x)
		{
			xBefore = (int)round(x / rx);
			xBefore = fmin(xBefore, imgWidth - 1);

			for (int c = 0; c < channel; ++c)
			{
				imgOut.at<Vec3b>(y, x)[c] = imgSrc.at<Vec3b>(yBefore, xBefore)[c];
			}
		}
	}
	imshow("imgSrc", imgSrc);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}