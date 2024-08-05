#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include<math.h>
#include "A_1_10.h"

using namespace cv;

void A8(void)
{
	printf_s("��ͼƬ���չ̶���С����ָ�����ڵ�����ֵȡ�������������ص����ֵ");
	Mat imgSrc = imread("C:\\Users\\Administrator\\Desktop\\img3.jpg");
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);
	int step = 8;//����
	int r = 8;//�ػ���Χ
	for (int y = 0; y < imgHeight; y += step)
	{
		for (int x = 0; x < imgWidth; x += step)
		{
			for (int c = 0; c < channel; c++)
			{
				//ͳ����r��Χ�ڵ������������ֵ
				int val = 0;
				for (int dy = 0; dy < r; dy++)
				{
					for (int dx = 0; dx < r; ++dx)
					{
						val = fmax(imgSrc.at<Vec3b>(y + dy, x + dx)[c],val);
					}
				}
				printf_s("%d", val);
				//��r��Χ����ͳһ��ֵ
				for (int dy = 0; dy < r; dy++)
				{
					for (int dx = 0; dx < r; ++dx)
					{
						imgOut.at<Vec3b>(y + dy, x + dx)[c] = (uchar)val;
					}
				}
			}
		}
	}

	imshow("src", imgSrc);
	imshow("out", imgOut);

	waitKey(0);
	destroyAllWindows();
}