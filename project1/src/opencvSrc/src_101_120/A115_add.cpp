#pragma once
#include "A_101_120_add.h"

void imgDeburring(cv::Mat src, cv::Mat& dst, int uThres, int vThres, cv::Size size, int dThres)
{
	dst = src.clone();
	int height = dst.rows;
	int width = dst.cols;
	int k;
	for (int i = 0; i < height - 1; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width - 1; j++)
		{
			//delete horizontal burrs 
			if (p[j] == 0 && p[j + 1] == 255)
			{
				if (j + uThres >= width)
				{
					for (int k = j + 1; k < width; k++)
						p[k] = 0;
				}
				else
				{
					for (k = j + 2; k <= j + uThres; k++)
					{
						if (p[k] == 0) break;
					}
					if (p[k] == 0)
					{
						for (int h = j + 1; h < k; h++)
							p[h] = 0;
					}
				}
			}
			if (p[j] == 255 && p[j + 1] == 0 && j <= uThres)
			{
				for (k = j; k >= 0; k--)
					p[k] = 0;
			}

			//delete vertical burrs  
			if (p[j] == 0 && p[j + width] == 255)
			{
				if (i + vThres >= height - 1)
				{
					for (k = j + width; k < j + (height - i) * width; k += width)
						p[k] = 0;
				}
				else
				{
					for (k = j + 2 * width; k <= j + vThres * width; k += width)
					{
						if (p[k] == 0) break;
					}
					if (p[k] == 0)
					{
						for (int h = j + width; h < k; h += width)
							p[h] = 0;
					}
				}
			}

			if (p[j] == 255 && p[j + width] == 0 && i <= vThres)
			{
				for (k = j; k >= j - i * width; k -= width)
					p[k] = 0;
			}
		}
	}

	if (size.height < 1)
		return;

	//blur to get smooth edge
	//blur(dst, dst, size);
	//threshold(dst, dst, uThres, 255, CV_THRESH_BINARY);
}

void imgAnalyze(Mat src, Mat& enSrc)
{
	Mat img_channels[3];
	split(src, img_channels);

	enSrc = img_channels[2].clone(); // imgMerge3.clone();

}

void copperAnalyze(Mat src, vector<Rect>& vRoi, Mat& srcInfo)
{
	Rect preRoi(1300, 0, 700, src.rows);
	Mat srcBinary = Mat::zeros(src.size(), CV_8UC1);
	src(preRoi).copyTo(srcBinary(preRoi));

	threshold(srcBinary, srcBinary, 250, 255, THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	//进行形态学操作
	morphologyEx(srcBinary, srcBinary, MORPH_ERODE, element);
	imgDeburring(srcBinary, srcBinary, 10, 10, Size(0, 0), 0);

	Mat tmp = srcBinary.clone();

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat srcShow, srcShow1;
	cvtColor(srcBinary, srcShow, CV_GRAY2BGR);
	cvtColor(src, srcShow1, CV_GRAY2BGR);

	vector<Rect> vcRect;
	vector<double> vAreas;
	int leftIdx(0), nIdx(0);

	for (size_t i = 0; i < contours.size(); i++)
	{
		double cArea = contourArea(contours[i]);

		if (cArea > 10000 && cArea < 50000)
		{
			Rect cRect = boundingRect(contours[i]);
			drawContours(srcShow, contours, (int)i, Scalar(255, 0, 0), 5, 8, hierarchy);
			rectangle(srcShow, cRect, Scalar(0, 255, 0), 1, 8, 0);

			vcRect.push_back(cRect);
			vAreas.push_back(cArea);

			leftIdx += cRect.x;
			nIdx++;
		}
	}
	if (nIdx == 0)
		return;

	sort(vcRect.begin(), vcRect.end(), compareMyVec);
	sort(vAreas.rbegin(), vAreas.rend());
	leftIdx /= nIdx;

	int vcSize = (int)vcRect.size();
	if (vcRect[vcSize - 1].height < 50)
		vcRect.erase(vcRect.begin() + vcSize - 1);
	if (vcRect[0].height < 50)
		vcRect.erase(vcRect.begin());

	vcSize = (int)vcRect.size();
	for (int i = 0; i < vcSize - 1; i++)
	{
		int y0 = (vcRect[i].y + vcRect[i].height / 2);
		int y1 = (vcRect[i + 1].y + vcRect[i + 1].height / 2);
		Rect ccRect(leftIdx, y0, 475, y1 - y0);
		Rect cccRect(leftIdx + 410, y0, 110, y1 - y0);
		rectangle(srcShow, ccRect, Scalar(0, 0, 255), 1, 8, 0);
		rectangle(srcShow, cccRect, Scalar(0, 255, 0), 1, 8, 0);
		rectangle(srcShow1, ccRect, Scalar(0, 0, 255), 1, 8, 0);
		rectangle(srcShow1, cccRect, Scalar(0, 255, 0), 1, 8, 0);
		vRoi.push_back(cccRect);
	}

	vector<Mat> vImgs;
	for (size_t i = 0; i < vRoi.size(); i++)
	{
		Mat croiImg = src(vRoi[i]).clone();
		vImgs.push_back(croiImg);
	}

	srcInfo = srcShow1.clone();
}