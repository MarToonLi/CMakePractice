#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"
#include <math.h>
using namespace cv;

void A10()
{
	printf_s("��ֵ�˲�\n");
	Mat imgSrc = imread("C:\\Users\\Administrator\\Desktop\\OpencvTestImg\\img512.png");
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);

	const int kernelSize = 3;//����˴�С
	int kernelRadius = floor(kernelSize / 2);//����˰뾶
	float kernelArray[kernelSize * kernelSize];//����˾���,�������

	Mat imgtemp = Mat::zeros(imgHeight + 2 * kernelRadius, imgWidth + 2 * kernelRadius, CV_8UC3);
	//����һ��ԭͼ��������ӱ߿�
	for (int y = 0; y < imgHeight; y++)
		for (int x = 0; x < imgWidth; x++)
			for (int c = 0; c < channel; c++)
				imgtemp.at<Vec3b>(y + kernelRadius, x + kernelRadius)[c] = imgSrc.at<Vec3b>(y, x)[c];

	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int c = 0; c < channel; ++c)
			{
				double val = 0;
				int count = 0;
				//��ֹ����
				for (int i = 0; i < kernelSize * kernelSize; i++)
					kernelArray[i] = 999;
				//�ھ���˷�Χ��ȡֵ
				for (int dy = -kernelRadius; dy < kernelRadius + 1; dy++)
					for (int dx = -kernelRadius; dx < kernelRadius + 1; dx++)
						if (((y + dy) >= 0) && ((x + dx) >= 0))
							kernelArray[count++] = (int)imgtemp.at<Vec3b>(y + dy, x + dx)[c];

				//��Ԫ�ؽ�������
				std::sort(kernelArray, kernelArray + (kernelSize * kernelSize));
				//ȡ��ֵ
				val = kernelArray[int(floor(count / 2)) + 1];
				imgOut.at<Vec3b>(y, x)[c] = (uchar)val;
			}
		}
	}

	imshow("src", imgSrc);
	imshow("out", imgOut);
	waitKey(0);
	destroyAllWindows();
}