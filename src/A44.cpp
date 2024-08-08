#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

// 霍夫变换
const int ANGLE_T = 180;//角度
const int RHO_MAX = 2448;

// 霍夫表
struct HoughTable {
	int table[RHO_MAX * 2][ANGLE_T];
};

//霍夫变换：投票
HoughTable Hough_vote(HoughTable hough_table, cv::Mat img)
{
	int height = img.rows;
	int width = img.cols;
	int rho = 0;
	double angle = 0;
	//遍历整个图片
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf_s("y:%d,x:%d\n",y,x);
			// 如果整个像素不是边，则直接跳过
			if (img.at<uchar>(y, x) != 255) {
				continue;
			}
			// 0 <= 角度 < 180
			for (int t = 0; t < ANGLE_T; t++) {
				angle = PI / 180 * t;
				//r = xcosθ + ysinθ
				rho = (int)(x * cos(angle) + y * sin(angle));
				//进行投票
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
	//imshow("canny",edge);

	////HoughTable hough_table;

	//// 初始化霍夫表
	////for (int rho = 0; rho < RHO_MAX * 2; rho++)
	////	for (int t = 0; t < ANGLE_T; t++)
	////		hough_table.table[rho][t] = 0;
	////printf_s("初始化结束\n");

	//// 霍夫投票
	////hough_table = Hough_vote(hough_table, edge);


		// 初始化霍夫变换参数
	int rho_res = 1; // 极径分辨率（像素）
	int theta_res = CV_PI / 45; // 极角分辨率（弧度）

	// 设置累加器数组大小
	int max_rho = max(img.rows * 2, img.cols * 2); // 最大可能的极径
	int max_theta = 180; // 极角范围 [0, 180)
	Mat accumulator = Mat::zeros(max_rho, max_theta, CV_32SC1); // 累加器数组初始化为0

	// 霍夫变换：对每个边缘像素进行投票
	for (int y = 0; y < edge.rows; ++y) {
		for (int x = 0; x < edge.cols; ++x) {
			if (edge.at<uchar>(y, x) > 0) { // 如果是边缘点
				for (int theta = 0; theta < max_theta; ++theta) {
					double rho = x * cos(theta * theta_res) + y * sin(theta * theta_res);
					int rho_idx = cvRound(rho) + max_rho / 2; // 将rho映射到累加器数组索引范围内

					// 边界检查和修正
					if (rho_idx >= 0 && rho_idx < accumulator.rows) {
						accumulator.at<int>(rho_idx, theta)++;
					}
				}
			}
		}
	}

	// 找到累加器数组中的峰值（即投票最多的直线）
	double threshold_votes = 50; // 设定阈值，过滤低于阈值的直线
	std::vector<Vec2i> lines; // 存储检测到的直线参数（rho, theta）

	for (int r = 0; r < accumulator.rows; ++r) {
		for (int t = 0; t < accumulator.cols; ++t) {
			if (accumulator.at<int>(r, t) > threshold_votes) {
				lines.push_back(Vec2i(r - max_rho / 2, t)); // 将rho和theta存储到lines中
			}
		}
	}

	// 在原始图像上绘制检测到的直线
	Mat result = img.clone();
	for (size_t i = 0; i < lines.size(); ++i) {
		double rho = lines[i][0];
		double theta = lines[i][1] * theta_res;
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(result, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA); // 绘制直线
	}




	waitKey(0);
	destroyAllWindows();
}