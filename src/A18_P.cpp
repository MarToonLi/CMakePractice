#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include<math.h>
#include "A_11_20.h"

using namespace cv;


void A18(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgGray = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC1);

	//�ȻҶȻ�
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.2126 * (float)imgSrc.at<cv::Vec3b>(y, x)[2] \
				+ 0.7152 * (float)img.at<cv::Vec3b>(y, x)[1] \
				+ 0.0722 * (float)img.at<cv::Vec3b>(y, x)[0];
		}
	}

	const int kSize = 3;
	//����뾶
	int kRadius = floor((double)kSize / 2);//����뾶

	double kEmboss[kSize][kSize] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2} };


	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			uchar val = 0;
			for (int dy = -kRadius; dy < kRadius + 1; dy++)
			{
				for (int dx = -kRadius; dx < kRadius + 1; dx++)
				{
					if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
					{
						val += imgGray.at<uchar>(y + dy, x + dx) * kEmboss[kRadius + dy][kRadius + dx];
					}

				}
			}
			imgOut.at<uchar>(y, x) = (uchar)val;

		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgOut", imgOut);;
	waitKey(0);
	destroyAllWindows();

}