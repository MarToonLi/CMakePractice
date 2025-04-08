#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Core>               // eigen 头文件必须放在 opencv 前面
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>
#include <chrono>
#include <future>
#include <cmath>
#include <fstream>


using namespace Eigen;

using namespace std;
using namespace cv;

/*
功能：深度图与点云图互转(产品存在倾斜)
*/

#define Gaussian_Size 20
#define Gaussian_Size_2 (Gaussian_Size>>1)

#pragma execution_character_set("utf-8") 

namespace NA112 {
	// 将深度图转换为点云，用于ROI图像
	bool D25toD30Way1(const cv::Mat& d25_image, float kong_min_valid_val, std::vector<cv::Point3f>& pixels_from_img, std::vector<cv::Point3f>& mms_from_img, std::vector<cv::Point3f>& second_mms_from_img, bool save_mark)
	{
		try {
			for (int j = 0; j < d25_image.cols; j++) {
				for (int i = 0; i < d25_image.rows; i++) {
					float z = static_cast<float>(d25_image.at<double>(i, j));
					cv::Point3f pixel_point(j, i, z);
					cv::Point3f mm_point(j * 0.05, i * 0.1, z * 1);   // 列 x  行y   高z
					pixels_from_img.push_back(pixel_point);
					mms_from_img.push_back(mm_point);

					if (z > 0) {
						second_mms_from_img.push_back(mm_point);
					}
				}
			}
		}
		catch (std::exception& e)
		{
			LOGE(e.what());
		}

		return true;
	}

	// 将深度图转换为点云，用于整张图像
	bool D25toD30Way2(const cv::Mat& d25_image, float kong_min_valid_val, std::vector<cv::Point3f>& pixels_from_img, std::vector<cv::Point3f>& mms_from_img, std::vector<cv::Point3f>& second_mms_from_img, bool save_mark)
	{
		float default_z = 47.0f;                                 // 因为图像像素基本上处于45左右
		float camera_distance_to_zeroplane = 106.5;              // 100 -> 91.57度；115 -> 88度；130 -> 83度；      


		try
		{
			/** 将深度图像素值修改为产品距离相机的真实距离 */
			cv::Mat resolutionMap, resolutionMapOri;
			cv::Mat d25_image_copy = d25_image.clone() + kong_min_valid_val;    // 产品距离相机零平面的距离
			cv::Rect roi(0, 0, d25_image_copy.cols, d25_image_copy.rows);
			d25_image_copy = camera_distance_to_zeroplane - d25_image_copy;     // 产品距离相机的距离
			d25_image_copy.convertTo(resolutionMapOri, CV_32FC1, 0.25, 16.25);  // 核心
			resolutionMap = resolutionMapOri(roi).clone();


			/** 修改极端异常数据的值 */
			for (int row = 0; row < resolutionMap.rows; row++)
			{
				for (int col = 0; col < resolutionMap.cols; col++)
				{
					float cr = resolutionMap.at<float>(row, col);
					if (cr > 100 || cr < 30) { resolutionMap.at<float>(row, col) = default_z; }
				}
			}


			/** 计算每个点分辨率的积分图 */
			cv::Mat rMat_integral;
			cv::integral(resolutionMap, rMat_integral, CV_32F);  //!? 积分图：某个坐标左上角所有像素值的总和。          

			ofstream outfile;
			if (save_mark) { outfile.open("F:\\Projects\\CMakePractice\\project1\\src\\opencvSrc\\src_101_120\\A112_vcloud_gn.txt", ios::out | ios::app); }


			/** 根据点分辨率的积分图修正横坐标 */
			for (int row = 0; row < resolutionMap.rows; row++)
			{
				for (int col = 0; col < resolutionMap.cols; col++)
				{
					float rz = d25_image.at<double>(row, col);      //!? 充当蒙版的作用

					float z = d25_image_copy.at<double>(row, col);
					float cy = row * 0.1;
					float cz = z * 1;
					// 计算当前行在X方向上的累积值，代表该行到当前列的累计宽度或长度
					// X坐标不是简单的列索引，而是基于某种累积度量（如实际物理宽度）的值。
					float cx = rMat_integral.at<float>(row + 1, col + 1) - rMat_integral.at<float>(row, col + 1); 
					cx *= 0.001f;                                   // X轴坐标转换（微米转毫米）

					if (cz > 1000 || cz < 100) { continue; }        // 去除异常数据

					cv::Point3f pixel_point(col, row, cz);
					cv::Point3f mm_point(cx, cy, cz);
					pixels_from_img.push_back(pixel_point);
					mms_from_img.push_back(mm_point);

					if (rz == 0) { continue; }                     // 去除异常数据
					second_mms_from_img.push_back(mm_point);

					if (save_mark) { outfile << cx << "," << cy << "," << cz << endl; }
				}
			}
			if (save_mark) { outfile.close(); }
		}
		catch (std::exception& e)
		{
			LOGE("e: {};", e.what());
		}

		return false;
	}

	// 将点向量改变成深度图
	void createDepthImage(const std::vector<Vector3d>& points, cv::Mat& depthImage) {

		for (const auto& point : points) {
			double x = point.x() / 0.05;
			double y = point.y() / 0.1;
			double z = point.z();

			if (z > -100) { // 确保z为正值
				int u = static_cast<int>(x);
				int v = static_cast<int>(y);

				// 检查u和v是否在图像范围内
				if (u >= 0 && u < depthImage.cols && v >= 0 && v < depthImage.rows) {
					depthImage.at<double>(v, u) = z;
				}
			}
		}
	}

    void A112_solver()
    {
		// 以一张倾斜的直角产品为例子；
		// 相机参数和型号。
		/**
		gofactor 2430
		x: 0.05;
		y: 0.1;
		z: 1;

		* X方向分辨率：0.037~0.057
		* 安装高度：75cm
		* Z方向视野范围：80cm
		* x方向分辨率 = alpha * 高度 + offset
		* 0.037 = alpha * 75 + offset
		* 0.057 = alpha * (75 + 80) + offset
		* alpha = 0.00025
		* offset = 18.25
		* 由于软件上可以调整参考平面，因此上述代码计算出的产品距离相机的距离并不准确，
		* 而这点偏差可以通过在90度产品上调整offset和camera_distance_to_zeroplane来完成！
		*/

		cv::Mat d25_img = cv::imread("F:\\Projects\\CMakePractice\\project1\\src\\opencvSrc\\src_101_120\\0_275TEST1_173133.tiff", IMREAD_ANYDEPTH);

		float kong_min_valid_val = -8;
		std::vector<cv::Point3f> pixels_from_img;
		std::vector<cv::Point3f> mms_from_img;
		std::vector<cv::Point3f> second_mms_from_img;
		bool save_mark;

		D25toD30Way2(d25_img, kong_min_valid_val, pixels_from_img, mms_from_img, second_mms_from_img, false);

        return;
    }
}
