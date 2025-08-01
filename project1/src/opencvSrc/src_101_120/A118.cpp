#include <omp.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>
#include <chrono>
#include <future>
#include <typeinfo>

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <utility>


#include <iostream>
#include <thread>
#include <vector>
#include <windows.h>
#include <queue>
#include <list>
#include <mutex>

#include <random>
#include <algorithm>  // std::sort
#include <numeric>    // std::accumulate
#include <cmath>      // std::round

using namespace std;
using namespace cv;
using namespace ml;

#pragma execution_character_set("utf-8") 



/**********************************
*
练习目标：
√ 1. 基于连通域的特征点的构建方式(Case1)
√ 2. 均值滤波后减去原图，类似黑帽运算，将边界提取出来(Case1)
3. 去除水印 https://zhuanlan.zhihu.com/p/340514505
4. 指标统计向量的二八比例均值技巧(Case3)
5. 顶帽和黑帽的应用场景(Case4)



=======================
传统图像处理常用方法：
1. （辅助）颜色空间转换：提取某个通道色彩或者增强某个物体的对比度
2. 形态学操作：去除噪声、提取边界（黑帽）、连接和分割对象；
3. （辅助）图像增强：灰度变换、直方图均衡、锐化等方法，增强目标物体间的对比度
4. 图像分割和特征提取：阈值分割、轮廓提取、分水岭算法（*），提升目标区域的对比度，将目标区域从图像中分割出来
5. （辅助）图像去噪：利用中值滤波去除椒盐噪声、通过傅里叶变换进行频域处理；
6. （优化）图像格式转换与优化：指针数据与Mat类型变量的转换
7. 特征点检测与匹配：（非工业视觉场景）

************************************/

// 1. 基于连通域的特征点的构建方式
namespace Case1 {
	const int FEATURE2_CONTRAST_DEGREE = 5;


	bool isfeature2(unsigned char* binary, int width, int height, int row, int col) {

		auto is_valid2 = [width, height](int index, int nr, int nc) {
			return (index >= 0) && (index <= width * height - 1) && (nc >= 0 && nc < width && nr >= 0 && nr < height);
			};

		for (int i = -2; i <= 2; ++i) {
			int nr = row; int nc = col + i;
			int index = nr * width + nc;
			int p = is_valid2(index, nr, nc) ? binary[index] : -1;
			if (p < FEATURE2_CONTRAST_DEGREE) { return false; }
		}

		return true;
	}



    void debug_get_feature2_img(cv::Mat& img, cv::Mat& img_bgr) {
		std::vector<int> feature2Columns;
		int width = img.cols;
		int height = img.rows;

#pragma omp parallel for num_threads(4)
		for (int col = 0; col < width; col++)
		{
			std::vector<int> feature2Points;
			feature2Points.reserve(height);  // 预分配内存

			// S1: 第一处理阶段
			for (int row = 0; row < height; row++) 
			{
				int p = img.at<uchar>(row, col);

				if (p > FEATURE2_CONTRAST_DEGREE && isfeature2(img.ptr(0), width, height, row, col))  // todo: 增加预先筛选的条件
				{
					feature2Points.push_back(row);
				}
			}


			//S2: 第二处理阶段
			int lineCount = 0;
			int lineCountThreshold = 3;
			const int lineGapThreshold = 2;

			if (!feature2Points.empty()) {   // 检测独立的白色线段（连续白像素视为一根线）
				lineCount = 1;
				for (size_t i = 1; i < feature2Points.size(); ++i) {
					if (feature2Points[i] - feature2Points[i - 1] > lineGapThreshold) {  // 非连续像素
						lineCount++;
					}
				}
			}

			if (lineCount >= lineCountThreshold) {
				#pragma omp critical
				feature2Columns.push_back(col);
			}  // 如果当前列有三根线，记录列索引

		}


		// S3: 第三处理
		if (feature2Columns.size() != 0)
		{
			int mark_height = 5;
			for (int i = 0; i < feature2Columns.size(); i++)
			{
				int col = feature2Columns[i];

				for (int j = 0; j < mark_height; j++)
				{
					img_bgr.at<cv::Vec3b>(0, col) = { 0, 255, 0 };
				}
			}
		}

    }


	void start() {
		// [输入]
		//std::string img_path = "H:\\Projects\\CMakePractice\\resources\\images\\A118_fenceng.bmp";  // 最佳案例：1200W分层图像
		std::string img_path = "H:\\Projects\\CMakePractice\\resources\\images\\A118_lvbo.png";  // 最佳案例：1200W分层图像



		// [操作]
		cv::Mat img_ori = cv::imread(img_path, cv::IMREAD_COLOR); 
		cv::Mat img_rst = img_ori.clone();
		cv::Mat img_gray, img_gray2, img_gray3;
		cv::cvtColor(img_ori, img_gray, cv::COLOR_BGR2GRAY);


		
		// [类似黑帽操作：分离比邻近点暗一些的斑块]
		//? 为什么水平核能将竖直型高像素值区域筛选出来？
		cv::blur(img_gray, img_gray2, cv::Size(21, 5)); 
		img_gray3 = img_gray2 - img_gray;
		/**
		* 1. 均值滤波使得，处于'小块横向型缺陷区域'的像素，其像素值'变化量'较小；而处于'小块竖直型缺陷区域'的像素，其像素值'变化量'较大;
		* 2. 而原图与均值滤波相减，则使得处于'小块横向型缺陷区域'的像素的像素值变成'变化量的值'，该结果中像素原来的像素值就不重要了！
		* 3. 理论推算核应该多大的问题：1） 其宽度必须大于常见的缺陷尺寸; 2) 尺寸越大计算量越大，因此宽度尺寸不能太大；3） 高度尺寸？
		* 4）滤波类操作本质上就是对小于核尺寸的缺陷区域进行有效的降噪，因此缺陷区域很大的话，滤波操作是无效的，因此我们分析一定是针对小型缺陷的！
		* 
		**/


		// [特征点：可能是分层区域内的像素点]
		// 特征：水平方向的4连通像素其必须都大于某个阈值
		debug_get_feature2_img(img_gray3, img_rst);
		/*
		* 1. opencv中的图像处理操作本质上都是遍历+连通域相关统计特征的结果！
		* 2. 所谓连通域特征其实就是被遍历像素点与其周围领域内像素点的数值和空间关系；
		* 3. 不要讲连通域特征想象得多么高大上！
		* 
		***/

		LOGD("Done.");
	}
}






namespace Case2 {


	// https://zhuanlan.zhihu.com/p/340514505
	// https://zhuanlan.zhihu.com/p/340514505
	// https://zhuanlan.zhihu.com/p/464252451


}














namespace Case3 {

	bool cal_target_val(std::vector<double> vals, float down = 0.8f, float up = 0.3f)
	{
		if (down <= up) { return false; }

		std::sort(vals.begin(), vals.end());
		// 2. 计算 20% 和 70% 的位置索引
		int n = vals.size();
		int start_index = static_cast<int>(std::round((1 - down) * n));
		int end_index = static_cast<int>(std::round((1 - up) * n));

		// 确保索引不越界
		start_index = (std::max)(0, start_index);
		end_index = (std::min)(n - 1, end_index);


		// 3. 提取子范围 [start_index, end_index]
		auto start_it = vals.begin() + start_index;
		auto end_it = vals.begin() + end_index + 1;  // 包含 end_index

		// 4. 计算均值
		double sum = std::accumulate(start_it, end_it, 0.0);
		int count = std::distance(start_it, end_it);
		double mean = (count > 0) ? sum / count : 0.0;

		LOGD("mean: {};", mean);
		LOGD("min: {};", vals.front());
		LOGD("max: {};", vals.back());

		return 1;
	}


	void start() {
		std::random_device rd;
		std::mt19937 gen(rd());

		// 定义分布范围 [10.0, 20.0)
		std::uniform_real_distribution<double> dist(10.0, 20.0);

		std::vector<double> random_floats;
		for (int i = 0; i < 100; ++i) {
			random_floats.push_back(dist(gen));
		}

		
		bool temp = cal_target_val(random_floats, 0.8, 0.3);


		/**
		* 去除噪声数据，提升估计值的重复计算的稳定性以及准确度！
		* 
		* 场景：
		* 1. 根据3D图像中铆钉有弧形的上表面深度值的集合，估计搞铆钉的高度
		* 2. 根据鲜花整个茎干的直径的集合，估计茎干的合理直径值
		* 
		**/


	}

}









namespace Case4 {



	/***
	*
	* 顶帽和黑帽的应用场景:
	* 1. 顶帽：本质上一句话，为了分离比邻近点亮一些的斑块；（原图 - 原图的开运算结果）
	* 2. 黑帽：本质上一句话，为了分离比邻近点暗一些的斑块；（原图的闭运算结果 - 原图）
	* 
	**/
}






namespace Case5 {
	/**
	* 1. 抵抗产品图像亮度不均匀的问题
	* 2. 不同颜色的产品如何统一处理获取产品轮廓
	* 3. 自适应直方图
	**/


	void start() {
		// 读取图像
		//cv::Mat image = cv::imread("H:\\Projects\\datasets\\AnomaDataset\\GXCF165ADV2\\abnormal\\Image_20250515151918604.bmp", IMREAD_GRAYSCALE);   //!? 缺陷样本
		cv::Mat image = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111 SamplesLibrary\\1200w\\Pic_2025_07_05_163638_1.bmp", IMREAD_GRAYSCALE);   //!? 亮暗不一


		// 操作方式： 原图像减去均值滤波结果
		cv::Mat image_clone1 = image.clone();

		cv::Mat image_clone1_blur1;
		cv::blur(image_clone1, image_clone1_blur1, cv::Size(21, 5));
		cv::Mat image_clone1_result1 = image_clone1_blur1 - image_clone1;   //!? 卷积核越大，平滑性越高，高频信息被平滑的幅度越大，则高频信息越明显(与周围像素的平均变化程度)

		cv::Mat image_clone1_blur2;
		cv::blur(image_clone1, image_clone1_blur2, cv::Size(41, 5));
		cv::Mat image_clone1_result2 = image_clone1_blur2 - image_clone1;

		cv::Mat image_clone1_blur3;
		cv::blur(image_clone1, image_clone1_blur3, cv::Size(61, 5));
		cv::Mat image_clone1_result3 = image_clone1_blur3 - image_clone1;  //!? 边界处由于其附近大多是白色像素，因此滤波核尺寸的增大基本不会影响该结果图像素值的变化


		// 中值滤波
		cv::Mat image_clone2 = image.clone();

		cv::Mat image_clone2_medianBlur1;
		cv::medianBlur(image_clone2, image_clone2_medianBlur1, 21);
		cv::Mat image_clone2_result1 = image_clone2_medianBlur1 - image_clone2;

		cv::Mat image_clone2_medianBlur2;
		cv::medianBlur(image_clone2, image_clone2_medianBlur2, 41);
		cv::Mat image_clone2_result2 = image_clone2_medianBlur2 - image_clone2;

		cv::Mat image_clone2_medianBlur3;
		cv::medianBlur(image_clone2, image_clone2_medianBlur3, 61);
		cv::Mat image_clone2_result3 = image_clone2_medianBlur3 - image_clone2;


		// 高斯滤波
		cv::Mat image_clone3 = image.clone();

		cv::Mat image_clone3_GaussianBlur1;
		cv::GaussianBlur(image_clone3, image_clone3_GaussianBlur1, cv::Size(21, 5), 0);
		cv::Mat image_clone3_result1 = image_clone3_GaussianBlur1 - image_clone3;

		cv::Mat image_clone3_GaussianBlur2;
		cv::GaussianBlur(image_clone3, image_clone3_GaussianBlur2, cv::Size(41, 5), 0);
		cv::Mat image_clone3_result2 = image_clone3_GaussianBlur2 - image_clone3;

		cv::Mat image_clone3_medianBlur3;
		cv::GaussianBlur(image_clone3, image_clone3_medianBlur3, cv::Size(61, 5), 0);
		cv::Mat image_clone3_result3 = image_clone3_medianBlur3 - image_clone3;

		cv::Mat image_clone3_medianBlur4;
		cv::GaussianBlur(image_clone3, image_clone3_medianBlur4, cv::Size(61, 5), 1);
		cv::Mat image_clone3_result4 = image_clone3_medianBlur4 - image_clone3;


		// 顶帽变换
		cv::Mat image_clone4 = image.clone();

		cv::Mat image_clone4_open1;
		cv::morphologyEx(image_clone4, image_clone4_open1, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 5)));
		cv::Mat image_clone4_result1 = image_clone4 - image_clone4_open1;

		cv::Mat image_clone4_open2;
		cv::morphologyEx(image_clone4, image_clone4_open2, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));
		cv::Mat image_clone4_result2 = image_clone4 - image_clone4_open2;

		cv::Mat image_clone4_open3;
		cv::morphologyEx(image_clone4, image_clone4_open3, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
		cv::Mat image_clone4_result3 = image_clone4 - image_clone4_open3;

		cv::Mat image_clone4_close1;
		cv::morphologyEx(image_clone4, image_clone4_close1, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
		cv::Mat image_clone4_result41 = image_clone4 - image_clone4_close1;
		cv::Mat image_clone4_result42 = image_clone4_close1 - image_clone4;

		cv::Mat image_clone4_close2;
		cv::morphologyEx(image_clone4, image_clone4_close2, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
		cv::Mat image_clone4_result51 = image_clone4 - image_clone4_close2;
		cv::Mat image_clone4_result52 = image_clone4_close2 - image_clone4;

		cv::Mat image_clone4_close3;
		cv::morphologyEx(image_clone4, image_clone4_close3, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));
		cv::Mat image_clone4_result61 = image_clone4 - image_clone4_close3;
		cv::Mat image_clone4_result62 = image_clone4_close3 - image_clone4;

		cv::Mat image_clone4_close4;
		cv::morphologyEx(image_clone4, image_clone4_close4, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 5)));
		cv::Mat image_clone4_result71 = image_clone4 - image_clone4_close4;
		cv::Mat image_clone4_result72 = image_clone4_close4 - image_clone4;    //!? 具有类似与均值滤波的效果！黑帽操作，分离周围暗一些的操作！

		cv::Mat image_clone4_close5;
		cv::morphologyEx(image_clone4, image_clone4_close5, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(41, 5)));
		cv::Mat image_clone4_result81 = image_clone4 - image_clone4_close5;
		cv::Mat image_clone4_result82 = image_clone4_close5 - image_clone4;    //!? 具有类似与均值滤波的效果！黑帽操作，分离周围暗一些的操作！


		cv::Mat image_clone4_close6;
		cv::morphologyEx(image_clone4, image_clone4_close6, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(41, 5)));
		cv::Mat image_clone4_result91 = image_clone4 - image_clone4_close6;
		cv::Mat image_clone4_result92 = image_clone4_close6 - image_clone4;    //!? 具有类似与均值滤波的效果！黑帽操作，分离周围暗一些的操作！



		// 自适应直方图均衡化
		cv::Mat image_clone5 = image.clone();
		cv::Mat image_clone5_result1;
		cv::Mat image_clone5_result2;
		cv::Mat image_clone5_result3;
		cv::Mat image_clone5_result4;
		cv::Mat image_clone5_result5;

		//!? clipLimit  限制每个小块直方图中单个灰度级的最大像素数，避免某些灰度级过度增强。
		//!? tileGridSize defines the number of tiles in row and column
		cv::Ptr<cv::CLAHE> clahe1 = cv::createCLAHE(20, cv::Size(10, 1));
		cv::Ptr<cv::CLAHE> clahe2 = cv::createCLAHE(20, cv::Size(10, 5));
		cv::Ptr<cv::CLAHE> clahe3 = cv::createCLAHE(20, cv::Size(10, 10));
		cv::Ptr<cv::CLAHE> clahe4 = cv::createCLAHE(20, cv::Size(1, 10));
		cv::Ptr<cv::CLAHE> clahe5 = cv::createCLAHE(20, cv::Size(5, 10));
		clahe1->apply(image_clone5, image_clone5_result1);
		clahe2->apply(image_clone5, image_clone5_result2);
		clahe3->apply(image_clone5, image_clone5_result3);
		clahe4->apply(image_clone5, image_clone5_result4);
		clahe5->apply(image_clone5, image_clone5_result5);


		// 查看一张图的RGB通道数据
		//!? 很大的前提时，你得有一张RGB图像才能做通道分离！
		//!? 合适的通道的判断条件：1） 能找到对应缺陷；2）便于你做各种运算，比如找到完整的产品边界！



		LOGD("mark");
	}







}










namespace NA118 {
    void A118_solver()
    {


        LOGD("......");
		Case5::start();

        return;
    }
}
