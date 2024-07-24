#pragma once
#include <opencv2/opencv.hpp>
#ifndef _A_41_50_
#define _A_41_50_

#define PI  atan(1) * 4

using namespace cv;

//Canny Function
//�ҶȻ�
Mat ImgGray(Mat img);
float Clip(float value, float min, float max);
//��˹�˲�
Mat GaussianFilter(Mat img, double Sigma);
//Sobel��ֱ
Mat SobelFilterV(Mat img);
//Sobelˮƽ
Mat SobelFilterH(Mat img);
Mat GetEdge(Mat imgX, Mat imgY);
Mat GetAngel(Mat imgX, Mat imgY);
//�Ǽ���ֵ���ƣ�NMS
Mat NMS(Mat angle, Mat edge);
//���ӱ�
Mat Histerisis(Mat edge, int HT, int LT);


void A41(Mat img);
void A42(Mat img);
void A43(Mat img);
void A44(Mat img);
void A45(Mat img);
void A46(Mat img);
void A47(Mat img);
void A48(Mat img);
void A49(Mat img);
void A50(Mat img);


#endif // !_A_41_50_