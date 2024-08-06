#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_1_10.h"

using namespace cv;

void A7_P(void)
{
	printf_s("将图片按照固定大小网格分割，网格内的像素值取网格内所有像素的平均值");
	//注意这里需要使用偶数尺寸的图片：128x128池化范围和步数应该满足倍数关系
	Mat imgSrc = imread("D:/Myself/MachineVision/resources/HaiLun-Apple/3-5/2024_08_02_15_34_22-10/DA2951215/ori/8__DA2951215.png");
	int imgHeight = imgSrc.rows;
	int imgWidth = imgSrc.cols;
	int channel = imgSrc.channels();

	int r = 8;    //池化范围; 池化计算时使用的数据量
	int step = 8; //步长； 与池化范围独立；默认是与池化范围相同
	Mat imgOut = Mat::zeros(imgHeight, imgWidth, CV_8UC3);

	for (int y = 0; y < imgHeight; y += step)
	{
		for (int x = 0; x < imgWidth; x += step)
		{
			for (int c = 0; c < channel; c++)
			{
				//统计在r范围内的所有像素之和
				int val = 0;
				for (int dy = 0; dy < r; dy++)
				{
					for (int dx = 0; dx < r; ++dx)
					{
						val += (double)imgSrc.at<Vec3b>(y + dy, x + dx)[c];
					}
				}

				//计算平均值
				val = val / (r * r);
				printf_s("%d", val);

				// 在r范围内所统一赋值
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
	imshow("out",imgOut);

	waitKey(0);
	destroyAllWindows();

}

