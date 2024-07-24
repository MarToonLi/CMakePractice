#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_51_60.h"
#include <math.h>
#include <time.h>

using namespace cv;

void A55(Mat img1,Mat img2)
{

	int imgHeight = img1.rows;
	int imgWeight = img1.cols;
	int channel = img1.channels();
	printf_s("channel:%d\n", channel);

	int imgh = img2.rows;
	int imgw = img2.cols;

	int val1 = 0, val2 = 0;
	int SAD = LONG_MAX;
	int _x = 0, _y = 0;//ƥ��ͼƬ����
	int X = 0, Y = 0;//ԭͼƥ������
	int diff = 0;
	bool flag = false;

	//����ֵ���:ԴͼƬ��ƥ��ͼƬ��������֮�Ȼ��ȡ����ֵ
	//ÿ�λ���һ������

	for (int y = 0; y < imgHeight - imgh; ++y)
	{
		for (int x = 0; x < imgWeight - imgw; ++x)
		{
			int sad = 0;
			Mat temp = img1.clone();

			Rect rect(x, y, imgw, imgh);//�������꣨x,y���;��εĳ�(x)��(y)
			cv::rectangle(temp, rect, Scalar(0, 0, 255), 1, LINE_AA, 0);
			cv::imshow("imgGray1", temp);
			cv::waitKey(5);

			//��ƥ��ͼƬ��Χ����SAD
			for (int _y = 0; _y < imgh; ++_y)
			{
				for (int _x = 0; _x < imgw; ++_x)
				{
					for (int c = 0; c < channel; ++c)
					{
						val1 = img1.at<Vec3b>(y + _y, x + _x)[c];
						val2 = img2.at<Vec3b>(_y, _x)[c];
						diff = val1 - val2;
						sad += abs(diff);
					}
				}
			}
			printf_s("ƥ�����꣺y:%d,x:%d SSD:%d\n", Y, X, sad);
			if (sad < SAD)
			{
				SAD = sad;
				Y = y;
				X = x;
				if (SAD == 0)
				{
					flag = true;
				}
			}
			if (flag)
				break;
		}
		if (flag)
			break;
	}

	Rect rect(X, Y, imgw, imgh);//�������꣨x,y���;��εĳ�(x)��(y)
	cv::rectangle(img1, rect, Scalar(0, 0, 255), 1, LINE_8, 0);

	printf_s("ƥ�����꣺y:%d,x:%d\nSSD:%d\n", Y, X, SAD);

	cv::imshow("img1", img1);
	cv::imshow("img2", img2);

	cv::waitKey(0);
	cv::destroyAllWindows();
}