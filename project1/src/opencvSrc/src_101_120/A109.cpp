#include <omp.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>
#include <chrono>

using namespace cv;

#pragma execution_character_set("utf-8") 

namespace NA109 {
	class ParallelAdd : public ParallelLoopBody//参考官方给出的answer，构造一个并行的循环体类
	{
	public:
		ParallelAdd(Mat& _src, Mat& _result)    //构造函数
		{
			src = _src;
			result = _result;

			src_data = src.ptr<uchar>(0);       // 数据起始指针
			step = src.step;                             //获取每一行的元素总个数（相当于cols*channels，等同于step1)
			result_data = _result.data;

		}

		void operator()(const Range& range) const //重载操作符（）
		{

			int step = (int)(result.step / result.elemSize1());
			// 遍历图像中心区域（排除边缘）
			for (int x = range.start; x < range.end - pad; ++x) {
				for (int y = pad; y < src.rows - pad; ++y) {
					// 收集邻域像素
					std::vector<uchar> window;
					window.reserve(kernelSize * kernelSize);

					for (int dy = -pad; dy <= pad; ++dy) {
						const uchar* p = src.ptr(y + dy);
						for (int dx = -pad; dx <= pad; ++dx) {
							window.push_back(p[x + dx]);
						}
					}

					// 快速获取中值
					auto mid = window.begin() + window.size() / 2;
					std::nth_element(window.begin(), mid, window.end());

					result_data[y * result.step + x] = *mid;
				}
			}
		}

	private:
		Mat src;
		Mat result;

		int kernelSize = 10;
		const int pad = kernelSize / 2;

		const uchar* src_data;
		uchar* result_data;
		int step;
	};


	void self_erode(cv::Mat& src, cv::Mat& imgErode)
	{
		imgErode = src.clone();

		int kernelSize = 10;
		const int pad = kernelSize / 2;

		// 遍历图像中心区域（排除边缘）
		for (int y = pad; y < src.rows - pad; ++y) {
			for (int x = pad; x < src.cols - pad; ++x) {
				// 收集邻域像素
				std::vector<uchar> window;
				window.reserve(kernelSize * kernelSize);

				for (int dy = -pad; dy <= pad; ++dy) {
					const uchar* p = src.ptr(y + dy);
					for (int dx = -pad; dx <= pad; ++dx) {
						window.push_back(p[x + dx]);
					}
				}

				// 快速获取中值
				auto mid = window.begin() + window.size() / 2;
				std::nth_element(window.begin(), mid, window.end());
				imgErode.at<uchar>(y, x) = *mid;
			}
		}
	}



	void self_erode_omp(cv::Mat& src, cv::Mat& imgErode)
	{
		imgErode = src.clone();

		int kernelSize = 10;
		const int pad = kernelSize / 2;

		int numThreads = omp_get_max_threads();

		for (int y = pad; y < src.rows - pad; ++y) {
			for (int x = pad; x < src.cols - pad; ++x) {
				// 收集邻域像素
				std::vector<uchar> window;
				window.reserve(kernelSize * kernelSize);

				for (int dy = -pad; dy <= pad; ++dy) {
					const uchar* p = src.ptr(y + dy);
					for (int dx = -pad; dx <= pad; ++dx) {
						window.push_back(p[x + dx]);
					}
				}

				// 快速获取中值
				auto mid = window.begin() + window.size() / 2;
				std::nth_element(window.begin(), mid, window.end());
				imgErode.at<uchar>(y, x) = *mid;
			}
		}


	}



	void test(Mat& _src1)
	{
		cv::Mat imgErode;
		int total_pics_num = 50;

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++)
		{
			self_erode(_src1, imgErode);
		}
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));
	}

	void test_omp(Mat& _src1)
	{
		cv::Mat imgErode;
		int total_pics_num = 50;

		auto start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < total_pics_num; i++)
		{
			self_erode_omp(_src1, imgErode);
		}
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));
	}


	void test_parallelfor(Mat& _src1)
	{
		cv::Mat imgErode = _src1.clone();
		int totalCols = _src1.cols;
		int total_pics_num = 50;

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++)
		{
			parallel_for_(Range(0, totalCols), ParallelAdd(_src1, imgErode), 100);  //隐式调用，并发
		}
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));
	}


    void A109_solver()
    {

		//cv::Mat imgSrc(720, 228, CV_8UC1); // 双精度浮点型
		//cv::randu(imgSrc, cv::Scalar(0), cv::Scalar(256)); // 上限不包含

        cv::Mat imgSrc = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\test_1562__ORI_DA2710107.jpg");

		cv::Mat gray;
		cv::cvtColor(imgSrc, gray, cv::COLOR_BGR2GRAY);

		//LOGD("imgSrc.width: {}; imgSrc.height: {}; imgSrc.type: {}ss;", imgSrc.cols, imgSrc.rows, imgSrc.type());


		//test(imgSrc);
		int numThreads = omp_get_max_threads();
		LOGD("numThreadsss: {};", numThreads);
		test_omp(imgSrc);
		//test_parallelfor(gray);

        return;
    }
}
