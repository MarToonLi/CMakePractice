#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

 //定义霍夫表的数据结构
typedef struct {
    cv::Mat table; // 使用cv::Mat存储霍夫表
} HoughTable;

// 定义霍夫表像素的结构
typedef struct {
    int rho;
    int theta;
    int vote;
} HoughPixel;

// 霍夫变换：非极大值抑制
HoughTable HoughNMS(const cv::Mat& hough_table, int RHO_MAX, int ANGLE_T, int N) {
    HoughTable output_hough_table;
    output_hough_table.table = cv::Mat::zeros(RHO_MAX, ANGLE_T, CV_32SC1); // 初始化霍夫表

    std::priority_queue<HoughPixel> top_pixels; // 使用优先队列来存储前 N 个最大像素

    // 遍历霍夫表，进行非极大值抑制
    for (int rho = 1; rho < RHO_MAX - 1; ++rho) {
        for (int t = 1; t < ANGLE_T - 1; ++t) {
            int value = hough_table.at<int>(rho, t);
            if (value == 0) continue; // 跳过无效像素

            bool is_max = true;

            // 检查八个方向的像素值
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dt = -1; dt <= 1; ++dt) {
                    if (dr == 0 && dt == 0) continue; // 跳过当前像素

                    if (value < hough_table.at<int>(rho + dr, t + dt)) {
                        is_max = false;
                        break;
                    }
                }
                if (!is_max) break;
            }

            if (is_max) {
                top_pixels.push({ rho, t, value });
                if (top_pixels.size() > N) top_pixels.pop(); // 维护前 N 个最大像素
            }
        }
    }

    // 将前 N 个最大像素放入输出霍夫表
    while (!top_pixels.empty()) {
        HoughPixel pixel = top_pixels.top();
        output_hough_table.table.at<int>(pixel.rho, pixel.theta) = pixel.vote;
        top_pixels.pop();
    }

    return output_hough_table;
}

void A45(Mat img)
{
	//Canny
	cv::Mat gray = ImgGray(img);
	cv::Mat gaussian = GaussianFilter(gray, 1.4);
	cv::Mat fy = SobelFilterV(gaussian);
	cv::Mat fx = SobelFilterH(gaussian);
	cv::Mat edge = GetEdge(fx, fy);
	cv::Mat angle = GetAngel(fx, fy);
	edge = NMS(angle, edge);
	edge = Histerisis(edge, 100, 30);
	imshow("canny", edge);

	//HoughTable hough_table;
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


    //HoughTable output_hough_table = HoughNMS(accumulator, max_rho, max_theta, 50);


    //std::vector<cv::Vec2f> lines2; // 存储检测到的直线参数 (rho, theta)

    //for (int rho = 0; rho < max_rho; ++rho) {
    //    for (int theta = 0; theta < max_theta; ++theta) {
    //        int vote = output_hough_table.table.at<int>(rho, theta);
    //        if (vote > 0) {
    //            lines2.push_back(cv::Vec2f(rho, theta)); // 存储直线参数 (rho, theta)
    //        }
    //    }
    //}


    //cv::Mat result_image = img.clone(); // 克隆原始图像作为结果图像

    //for (size_t i = 0; i < lines2.size(); ++i) {
    //    float rho = lines2[i][0];
    //    float theta = lines2[i][1];

    //    // 将极坐标转换为直线表示
    //    double a = cos(theta);
    //    double b = sin(theta);
    //    double x0 = a * rho;
    //    double y0 = b * rho;

    //    // 计算直线上的两个点
    //    cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
    //    cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

    //    // 绘制直线
    //    cv::line(result_image, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    //}




	waitKey(0);
	destroyAllWindows();
}