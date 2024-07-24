#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

// ����任
const int ANGLE_T = 180;//�Ƕ�
const int RHO_MAX = 255;

// �����
struct HoughTable {
	int table[RHO_MAX * 2][ANGLE_T];
};

//����任��ͶƱ
HoughTable Hough_vote(HoughTable hough_table, cv::Mat img)
{
	int height = img.rows;
	int width = img.cols;
	int rho = 0;
	double angle = 0;
	//��������ͼƬ
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf_s("y:%d,x:%d\n",y,x);
			// ����������ز��Ǳߣ���ֱ������
			if (img.at<uchar>(y, x) != 255) {
				continue;
			}
			// 0 <= �Ƕ� < 180
			for (int t = 0; t < ANGLE_T; t++) {
				angle = PI / 180 * t;
				//r = xcos�� + ysin��
				rho = (int)(x * cos(angle) + y * sin(angle));
				//����ͶƱ
				hough_table.table[rho + RHO_MAX][t] ++;
			}
		}
	}
	return hough_table;
}

void A44(Mat img)
{
	//Canny
	// BGR -> Gray
	cv::Mat gray = ImgGray(img);

	// gaussian filter
	cv::Mat gaussian = GaussianFilter(gray, 1.4);

	// sobel filter (vertical)
	cv::Mat fy = SobelFilterV(gaussian);

	// sobel filter (horizontal)
	cv::Mat fx = SobelFilterH(gaussian);

	// get edge
	cv::Mat edge = GetEdge(fx, fy);

	// get angle
	cv::Mat angle = GetAngel(fx, fy);

	// edge non-maximum suppression
	edge = NMS(angle, edge);

	edge = Histerisis(edge, 100, 30);

	//cv::imshow("gray", gray);
	//cv::imshow("gaussian", gaussian);
	//cv::imshow("fy", fy);
	//cv::imshow("fx", fx);
	imshow("canny",edge);

	HoughTable hough_table;

	// ��ʼ�������
	for (int rho = 0; rho < RHO_MAX * 2; rho++)
		for (int t = 0; t < ANGLE_T; t++)
			hough_table.table[rho][t] = 0;
	printf_s("��ʼ������\n");

	// ����ͶƱ
	hough_table = Hough_vote(hough_table, edge);

	waitKey(0);
	destroyAllWindows();
}