#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include<math.h>
#include "A_11_20.h"

using namespace cv;


void A11(Mat img)
{
	/*
	ԭ��
	�����˷�Χ���������صľ�ֵ��
	ͼƬ��СӦ�����㲽����������
	*/

	Mat imgSrc = img;

	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);


	//����˴�С
	const int kSize = 3;
	//����˰뾶
	int kRadius = floor((double)kSize / 2);
	Mat imgtemp = Mat::zeros(imgHeight + 2 * kRadius, imgWidth + 2 * kRadius, CV_8UC3);
	printf("kRadius:%d\n", kRadius);
	//����˾���
	int kArray[kSize * kSize];
	int count = 0;
	/*��ֵ�˲������
	[1,1,1]
	[1,1,1]
	[1,1,1]
	*/
	//���þ����,���������������û���õ���Ϊ�˿��ټ��㣬ֱ����sum/n;
	for (int i = 0; i < kSize * kSize; ++i)
	{
		kArray[i] = 1;
	}
	//����һ��ԭͼ��������ӱ߿�
	for (int y = 0; y < imgHeight; y++)
		for (int x = 0; x < imgWidth; x++)
			for (int c = 0; c < channel; c++)
				imgtemp.at<Vec3b>(y + kRadius, x + kRadius)[c] = imgSrc.at<Vec3b>(y, x)[c];

	for (int y = 0; y < imgHeight; y++)
	{
		for (int x = 0; x < imgWidth; x++)
		{
			//printf_s("y=%d,x=%d\n", y, x);
			for (int c = 0; c < channel; c++)
			{
				count = 0;
				double val = 0;//�洢����˰뾶���������غ�
				for (int dy = -kRadius; dy < kRadius + 1; dy++) {
					for (int dx = -kRadius; dx < kRadius + 1; dx++) {
						if (((y + dy) >= 0) && ((x + dx) >= 0)) {
							//printf("%d+%d,%d+%d\n", y, dy, x, dx);
							val += (int)imgtemp.at<cv::Vec3b>(y + dy, x + dx)[c];
							//printf_s("yes\n");
						}
					}
				}
				//���ֵ
				val = val / (kSize * kSize);
				imgOut.at<Vec3b>(y, x)[c] = (uchar)val;
			}
		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgOut", imgOut);
	waitKey(0);
	destroyAllWindows();
}