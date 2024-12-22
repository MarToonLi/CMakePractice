#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "A_1_10.h"

using namespace cv;



void functionA4_1(cv::Mat& imgSrc) {

	int imgHeight = imgSrc.rows;
	int imgWeight = imgSrc.cols;

	Mat imgGray = Mat::zeros(imgHeight, imgWeight, CV_8UC1);
	Mat imgOut = Mat::zeros(imgHeight, imgWeight, CV_8UC1);

	//灰度化
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.114 * (float)imgSrc.at<Vec3b>(y, x)[2]
				+ 0.587 * (float)imgSrc.at<Vec3b>(y, x)[1]
				+ 0.299 * (float)imgSrc.at<Vec3b>(y, x)[0];
		}
	}

	const int grayScale = 256;//灰度值

	int pixelCount[grayScale] = { 0 };//灰度直方图
	float pixelPro[grayScale] = { 0 };//各个灰度值占总体的比例

	double w0, w1; // 背景/前景 像素占比
	double u0, u1; // 前景/背景 平均灰度值
	double p0, p1;
	double g = 0;  // 类间方差

	double max_g = 0;//最大类间方差
	double good_k = 0;//最优阈值
	int pixelSum = imgHeight * imgWeight;//总像素值

	//统计图片中各个灰度值的个数
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			int val = imgGray.at<uchar>(y, x);
			pixelCount[val]++;
		}
	}

	//统计图片中各个灰度值所占的比例
	for (int i = 0; i < grayScale; ++i)
	{
		pixelPro[i] = 1.0 * pixelCount[i] / pixelSum;
	}

	//k:暂定阈值(0-255)
	for (int k = 0; k < grayScale; ++k)
	{
		w0 = w1 = u0 = u1 = g = 0;
		p0 = p1 = 0;

		//前景，背景区分 [0-k]  [k+1-255]
		for (int i = 0; i < grayScale; ++i)
		{
			//如果当前像素值小于阈值k则属于背景，反之属于前景
			if (i <= k)
			{
				//计算背景像素占比
				w0 += pixelPro[i];
				//计算当前灰度值发生的概率:灰度值*灰度值发生的概率
				p0 += (i * pixelPro[i]);

			}
			else
			{
				//计算前景像素占比
				w1 += pixelPro[i];
				p1 += (i * pixelPro[i]);
			}
		}


		//! 计算平均灰度值：p0/w0
		u0 = p0 / w0;
		u1 = p1 / w1;


		//计算类内方差
		g = (float)(w0 * w1 * pow((u0 - u1), 2));
		if (g > max_g)
		{
			max_g = g;
			good_k = k;

		}
	}
	printf_s("good k;%f\n", good_k);

	//取得最好的k值，以k值作为阈值进行二值化
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			if (imgGray.at<uchar>(y, x) > good_k)
				imgOut.at<uchar>(y, x) = 255;
			else
				imgOut.at<uchar>(y, x) = 0;
		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgGray", imgGray);
	imshow("ostu", imgOut);
	//imshow("API", API);
	waitKey(0);
	destroyAllWindows();
}



void functionA4_2(cv::Mat& imgSrc) {

	int imgHeight = imgSrc.rows;
	int imgWeight = imgSrc.cols;

	Mat imgGray = Mat::zeros(imgHeight, imgWeight, CV_8UC1);
	Mat imgOut = Mat::zeros(imgHeight, imgWeight, CV_8UC1);

	//灰度化
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			imgGray.at<uchar>(y, x) = 0.114 * (float)imgSrc.at<Vec3b>(y, x)[2]
				+ 0.587 * (float)imgSrc.at<Vec3b>(y, x)[1]
				+ 0.299 * (float)imgSrc.at<Vec3b>(y, x)[0];
		}
	}

	const int grayScale = 256;//灰度值

	int pixelCount[grayScale] = { 0 };//灰度直方图
	float pixelPro[grayScale] = { 0 };//各个灰度值占总体的比例


	double w0, w1, w_total; // 背景像素总和在整图中的占比；  前景像素总和在整图中的占比
	double u0, u1, u_total; // 背景灰度值均值 ； 前景灰度值均值
	double p0, p1, p_total; // 背景灰度值均值 ； 前景灰度值均值

	double g = 0;  // 类间方差

	double max_g = 0;//最大类间方差
	double good_k = 0;//最优阈值
	int pixelSum = imgHeight * imgWeight;//总像素值

	//统计图片中各个灰度值的个数
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			int val = imgGray.at<uchar>(y, x);
			pixelCount[val]++;
		}
	}

	//统计图片中各个灰度值所占的比例
	for (int i = 0; i < grayScale; ++i)
	{
		pixelPro[i] = 1.0 * pixelCount[i] / pixelSum;
	}

	//k:暂定阈值(0-255)
	for (int k = 0; k < grayScale; ++k)
	{
		w0 = w1 = w_total = u0 = u1 = u_total = g = 0;
		p0 = p1 = p_total = 0;

		//前景，背景区分 [0-k]  [k+1-255]
		for (int i = 0; i < grayScale; ++i)
		{
			//如果当前像素值小于阈值k则属于背景，反之属于前景
			if (i <= k)
			{
				//计算背景像素占比
				w0 += pixelPro[i];
				//计算当前灰度值发生的概率:灰度值*灰度值发生的概率
				p0 += (i * pixelPro[i]);

			}
			else
			{
				//计算前景像素占比
				w1 += pixelPro[i];
				p1 += (i * pixelPro[i]);
			}

			w_total += pixelPro[i];
			p_total += (i * pixelPro[i]);
		}




		//! 计算平均灰度值：p0/w0
		u0 = p0 / w0;
		u1 = p1 / w1;
		u_total = p_total / w_total;


		//计算类内方差
		g = (float)(w0 * w1 * pow((u0 - u1), 2));
		//g = (float)(w0 * pow((u0 - u_total), 2) + w1 * pow(u1 - u_total,2));

		std::cout << "p0: " << p0 << "; p1: " << p1 << "; p_total: " << p_total << "; g: " << g <<std::endl;
		if (g > max_g)
		{
			max_g = g;
			good_k = k;

		}
	}
	printf_s("good k;%f\n", good_k);

	//取得最好的k值，以k值作为阈值进行二值化
	for (int y = 0; y < imgHeight; ++y)
	{
		for (int x = 0; x < imgWeight; ++x)
		{
			if (imgGray.at<uchar>(y, x) > good_k)
				imgOut.at<uchar>(y, x) = 255;
			else
				imgOut.at<uchar>(y, x) = 0;
		}
	}

	imshow("imgSrc", imgSrc);
	imshow("imgGray", imgGray);
	imshow("ostu", imgOut);
	//imshow("API", API);
	waitKey(0);
	destroyAllWindows();
}










void A4(void)
{
	printf_s("大津二值化算法");
	Mat imgSrc = imread("D:/Myself/MachineVision/resources/HaiLun-Apple/3-5/2024_08_02_15_34_22-10/DA2951215/ori/8__DA2951215.png");
	//Mat API = imread("C:\\Users\\Administrator\\Desktop\\img1.jpg");


	functionA4_2(imgSrc);
}