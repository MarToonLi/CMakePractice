#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include<math.h>
#include "A_11_20.h"

using namespace cv;


void A15(Mat img)
{
	Mat imgSrc = img;
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgGray = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	Mat imgOutH = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	Mat imgOutV = Mat::zeros(imgHeight, imgWidth, CV_8UC1);

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
	//�����
	//���þ����:ˮƽ
	double kSobelHorizontal[kSize][kSize] = { {1, 0, 1}, 
											{2, 0, -2}, 
											{-1, 0, -1} };

	//���þ����:��ֱ
	double kSobelVertical[kSize][kSize] = { {1, 2, 1}, 
											{0, 0, 0}, 
											{-1, -2, -1} };

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			uchar valH = 0;
			uchar valV = 0;
			for (int dy = -kRadius; dy < kRadius + 1; dy++)
			{
				for (int dx = -kRadius; dx < kRadius + 1; dx++)
				{
					if (((y + dy >= 0)) && (x + dx) >= 0 && ((y + dy) < imgHeight) && ((x + dx) < imgWidth))
					{
						valH += imgGray.at<uchar>(y + dy, x + dx) * kSobelHorizontal[kRadius + dy][kRadius + dx];
						valV += imgGray.at<uchar>(y + dy, x + dx) * kSobelVertical[kRadius + dy][kRadius + dx];
					}

				}
			}
			imgOutH.at<uchar>(y, x) = (uchar)valH;
			imgOutV.at<uchar>(y, x) = (uchar)valV;

		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgOutH", imgOutH);
	imshow("imgOutV", imgOutV);
	waitKey(0);
	destroyAllWindows();

}