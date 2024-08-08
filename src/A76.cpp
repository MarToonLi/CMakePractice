#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_71_80.h"
using namespace cv;

Mat GaussianPyramid(Mat img, float n)
{
	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int channel = img.channels();
	//缩小
	double rx;
	double ry;
	rx = ry = n;
	//缩放后的宽高
	int resizeHeight = (int)(imgHeight * ry);
	int resizeWidth = (int)(imgWidth * rx);
	Mat imgOut = Mat::zeros(resizeHeight, resizeWidth, CV_8UC1);
	//计算值
	int xBefore, yBefore;
	double dy, dx;
	double val;
	for (int y = 0; y < resizeHeight; ++y)
	{
		yBefore = (int)floor(y / ry);
		yBefore = fmin(yBefore, imgHeight - 1);
		dy = y / ry - yBefore;

		for (int x = 0; x < resizeWidth; ++x)
		{
			xBefore = (int)floor(x / rx);
			xBefore = fmin(xBefore, imgWidth - 1);
			dx = x / rx - xBefore;

			for (int c = 0; c < channel; ++c)
			{
				val = (1. - dx) * (1. - dy) * img.at<uchar>(yBefore, xBefore) +
					dx * (1. - dy) * img.at<uchar>(yBefore, MIN(xBefore + 1, imgWidth - 1)) +
					(1. - dx) * dy * img.at<uchar>(MIN(yBefore + 1, imgHeight - 1), xBefore) +
					dx * dy * img.at<uchar>(MIN(yBefore + 1, imgHeight - 1), xBefore);
				imgOut.at<uchar>(y, x) = (uchar)val;
			}
		}
	}
	//放大
	return imgOut;
}


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to compute Gaussian pyramid
void buildGaussianPyramid(Mat& src, vector<Mat>& pyramid, int levels) {
	Mat current = src.clone();
	pyramid.push_back(current);
	for (int i = 0; i < levels; ++i) {
		Mat next;
		pyrDown(current, next);
		pyramid.push_back(next);
		current = next;
	}
}

// Function to compute DoG (Difference of Gaussians) pyramid
void buildDoGPyramid(vector<Mat>& gaussianPyramid, vector<Mat>& dogPyramid) {
	for (int i = 1; i < gaussianPyramid.size(); ++i) {
		Mat dog;
		subtract(gaussianPyramid[i - 1], gaussianPyramid[i], dog);
		dogPyramid.push_back(dog);
	}
}

// Function to compute saliency map from DoG pyramid
Mat computeSaliencyMap(vector<Mat>& dogPyramid) {
	Mat saliencyMap = Mat::zeros(dogPyramid[0].size(), CV_32F);
	for (int i = 0; i < dogPyramid.size(); ++i) {
		Mat dog = dogPyramid[i];
		Mat squaredDog;
		multiply(dog, dog, squaredDog); // Element-wise square of DoG

		cv::Mat squaredDog_32S;
		squaredDog.convertTo(squaredDog_32S, CV_32F);

		saliencyMap = saliencyMap + squaredDog_32S;
		//saliencyMap += squaredDog_32S; // Accumulate squared DoG maps
	}
	// Normalize saliency map to range [0, 1]
	normalize(saliencyMap, saliencyMap, 0, 1, NORM_MINMAX);
	return saliencyMap;
}

int vvv(cv::Mat image) {
	// Read input image
	//Mat image = imread("input.jpg", IMREAD_COLOR);
	//if (image.empty()) {
	//	cerr << "Error: Couldn't open the image file." << endl;
	//	return -1;
	//}

	//// Convert image to grayscale
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// Parameters for Gaussian pyramid and DoG pyramid
	const int numLevels = 5; // Number of levels in pyramid
	vector<Mat> gaussianPyramid;
	vector<Mat> dogPyramid;

	// Build Gaussian pyramid
	buildGaussianPyramid(grayImage, gaussianPyramid, numLevels);


	// Ensure all Gaussian pyramid levels have the same size
	for (int i = 1; i < gaussianPyramid.size(); ++i) {
		if (gaussianPyramid[i].size() != gaussianPyramid[0].size()) {
			resize(gaussianPyramid[i], gaussianPyramid[i], gaussianPyramid[0].size());
		}
	}

	// Build DoG pyramid
	buildDoGPyramid(gaussianPyramid, dogPyramid);

	// Compute saliency map
	Mat saliencyMap = computeSaliencyMap(dogPyramid);

	// Display or further process saliency map
	imshow("Saliency Map", saliencyMap);
	waitKey(0);

	return 0;
}





void A76(Mat img)
{

	vvv(img);

	//int imgHeight = img.rows;
	//int imgWidth = img.cols;
	//int channel = img.channels();
	//Mat imgGray = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	////灰度化
	//for (int y = 0; y < imgHeight; ++y)
	//{
	//	for (int x = 0; x < imgWidth; ++x)
	//	{
	//		imgGray.at<uchar>(y, x) = 0.2126 * (float)img.at<Vec3b>(y, x)[2] \
	//			+ 0.7152 * (float)img.at<Vec3b>(y, x)[1] \
	//			+ 0.0722 * (float)img.at<Vec3b>(y, x)[0];
	//	}
	//}

	////使用vecto进行图片存储
	//int n = 6;
	//Mat imgOut[6];
	//char name[6][5] = { "out1","out2","out3","out4","out5","out6" };

	//for (int i = 0; i < n; i++)
	//{
	//	Mat out = GaussianPyramid(imgGray, pow(0.5, i));
	//	out = GaussianPyramid(out, pow(2, i));
	//	imgOut[i] = out;
	//}
	////display
	//imgHeight = imgOut[0].rows;
	//imgWidth = imgOut[0].cols;
	//Mat res = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
	////imshow(name[i], imgOut[i]);
	////(0, 1)、(0, 3)、(0, 5)、(1, 4)、(2, 3)、(3, 5)

	//for (int y = 0; y < imgHeight; ++y)
	//{
	//	for (int x = 0; x < imgWidth; ++x)
	//	{
	//		int val = 0;
	//		val += std::abs(imgOut[0].at<uchar>(y, x) - imgOut[1].at<uchar>(y, x));
	//		val += std::abs(imgOut[0].at<uchar>(y, x) - imgOut[3].at<uchar>(y, x));
	//		val += std::abs(imgOut[0].at<uchar>(y, x) - imgOut[5].at<uchar>(y, x));
	//		val += std::abs(imgOut[1].at<uchar>(y, x) - imgOut[4].at<uchar>(y, x));
	//		val += std::abs(imgOut[2].at<uchar>(y, x) - imgOut[3].at<uchar>(y, x));
	//		val += std::abs(imgOut[3].at<uchar>(y, x) - imgOut[5].at<uchar>(y, x));
	//		res.at<uchar>(y, x) = val;
	//	}
	//}

	//imshow("res", res);
	//imshow("imgGray", imgGray);
	//waitKey(0);
	//destroyAllWindows();
}