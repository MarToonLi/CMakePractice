#pragma once
#include <opencv2/opencv.hpp>
#include "logger.h"
#include <algorithm>
#include "A_101_120_add.h"
#include <windows.h>  // 获取memory
#include <psapi.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

#ifndef _A_101_120_
#define _A_101_120_

#define PI  atan(1) * 4

using namespace cv;
using namespace std;

void A101();
void A102();
void A103();
void A104();
void A105();

namespace NA106 {
	void A106();
}

namespace NA107 {
	void A107();
}

namespace NA108 {
	void A108_solver();
}


namespace NA109 {
	void A109_solver();
}


namespace NA110 {
	void A110_solver();
}

namespace NA111 {
	int picshadowx(cv::Mat binary, cv::Mat* show);
	void doing(cv::Mat imgori);
	void A111_solver();
}


namespace NA112 {
	void A112_solver();
}

namespace NA113 {
	void A113_solver();
}


namespace NA114 {

	//IncrementalMean local_meaner;


	void A114_solver();

    struct SingleMat
    {
        int camPos = -1;
        std::chrono::steady_clock::time_point starttime;
        int w, h, format;
        char* imgori = nullptr;
        cv::Mat imgrst;
        size_t index;
        int groupsize;
        int _iresult;
        bool _bshow = true;
        std::string sn_fromscanner;
        std::string cam_serial;
        std::vector<std::string> error_message;

        // 默认构造函数
        SingleMat() = default;   // 相当于 SingleMat() {}；

        // 拷贝构造函数（深度拷贝）
        SingleMat(const SingleMat& other)
            : camPos(other.camPos),
            starttime(other.starttime),
            w(other.w),
            h(other.h),
            format(other.format),
            imgrst(other.imgrst.clone()), // 深度拷贝 cv::Mat
            index(other.index),
            groupsize(other.groupsize),
            _iresult(other._iresult),
            _bshow(other._bshow),
            sn_fromscanner(other.sn_fromscanner),
            cam_serial(other.cam_serial),
            error_message(other.error_message)
        {
            // 其他成员函数通过调用其所属数据类型的拷贝构造函数，实现深度拷贝；
            // 深度拷贝 imgori
            if (other.imgori != nullptr && other.w > 0 && other.h > 0)
            {
                size_t size = other.w * other.h * (other.format == 1 ? 1 : 3);
                imgori = new char[size];
                std::memcpy(imgori, other.imgori, size);
            }
            else
            {
                imgori = nullptr;
            }
        }

        // 拷贝赋值运算符（深度拷贝）
        SingleMat& operator=(const SingleMat& other)
        {
            if (this != &other)
            {
                // 释放原有资源
                delete[] imgori;

                // 拷贝基本类型成员
                camPos = other.camPos;
                starttime = other.starttime;
                w = other.w;
                h = other.h;
                format = other.format;
                index = other.index;
                groupsize = other.groupsize;
                _iresult = other._iresult;
                _bshow = other._bshow;

                // 拷贝字符串和容器
                sn_fromscanner = other.sn_fromscanner;
                cam_serial = other.cam_serial;
                error_message = other.error_message;

                // 深度拷贝 cv::Mat
                imgrst = other.imgrst.clone();

                // 深度拷贝 imgori
                if (other.imgori != nullptr && other.w > 0 && other.h > 0)
                {
                    size_t size = other.w * other.h * (other.format == 1 ? 1 : 3);
                    imgori = new char[size];
                    std::memcpy(imgori, other.imgori, size);
                }
                else
                {
                    imgori = nullptr;
                }
            }
            return *this;
        }

        // 移动构造函数
        SingleMat(SingleMat&& other) noexcept  // 移动构造函数，通过窃取临时对象（右值）的资源来构造新对象的构造函数，同时将原对象重置。
            : camPos(other.camPos),
            starttime(other.starttime),
            w(other.w),
            h(other.h),
            format(other.format),
            imgori(other.imgori),
            imgrst(std::move(other.imgrst)),
            index(other.index),
            groupsize(other.groupsize),
            _iresult(other._iresult),
            _bshow(other._bshow),
            sn_fromscanner(std::move(other.sn_fromscanner)),
            cam_serial(std::move(other.cam_serial)),
            error_message(std::move(other.error_message))
        {
            other.imgori = nullptr;
        }

        // 移动赋值运算符
        SingleMat& operator=(SingleMat&& other) noexcept
        {
            if (this != &other)
            {
                delete[] imgori;

                camPos = other.camPos;
                starttime = other.starttime;
                w = other.w;
                h = other.h;
                format = other.format;
                imgori = other.imgori;
                imgrst = std::move(other.imgrst);
                index = other.index;
                groupsize = other.groupsize;
                _iresult = other._iresult;
                _bshow = other._bshow;
                sn_fromscanner = std::move(other.sn_fromscanner);
                cam_serial = std::move(other.cam_serial);
                error_message = std::move(other.error_message);

                other.imgori = nullptr;
            }
            return *this;
        }

        ~SingleMat()
        {
            delete[] imgori;
        }
    };
}


namespace NA115 {
    void A115_solver();
}


namespace NA116 {
    void A116_solver();
}


namespace NA117 {
    void A117_solver();
}

namespace NA118 {
    void A118_solver();
}

namespace NA119 {

	//!? 存储模板匹配的相关数据和状态，可能是用于多尺度金字塔匹配或者某种优化的模板匹配算法比如归一化互相关。
	struct s_TemplData
	{
		vector<Mat> vecPyramid;         // 存储不同尺度的模板图像金字塔
		vector<Scalar> vecTemplMean;    // 每个金字塔层模板的均值（BGR或者灰度）
		vector<double> vecTemplNorm;    // 每个金字塔模板的范数（用于归一化计算）
		vector<double> vecInvArea;      // 每个金字塔层模板的逆面积，用于快速计算
		vector<BOOL> vecResultEqual1;   // 标记每一层匹配结果是否完全匹配（用于提前终止），如果某层匹配结果完全一致，侧跳过后续层计算
		BOOL bIsPatternLearned;         // 标记模板是否已经预处理，比如金字塔、均值等信息是否计算完毕
		int iBorderColor;               // 模板边界的填充颜色，用于金字塔下采样时的边界处理。


		void clear()  // 清空并释放内存，比vec.clear更彻底
		{
			vector<Mat>().swap(vecPyramid);
			vector<double>().swap(vecTemplNorm);
			vector<double>().swap(vecInvArea);
			vector<Scalar>().swap(vecTemplMean);
			vector<BOOL>().swap(vecResultEqual1);
		}
		void resize(int iSize)   // 调整容器大小；构建金字塔时预分配空间，避免动态扩容的开销；
		{
			vecTemplMean.resize(iSize);
			vecTemplNorm.resize(iSize, 0);  // 默认填充值
			vecInvArea.resize(iSize, 1);
			vecResultEqual1.resize(iSize, FALSE);
		}
		s_TemplData()
		{
			bIsPatternLearned = FALSE;
		}
	};


	struct s_MatchParameter
	{
		Point2d pt;
		double dMatchScore;
		double dMatchAngle;
		//Mat matRotatedSrc;
		Rect rectRoi;
		double dAngleStart;
		double dAngleEnd;
		RotatedRect rectR;
		Rect rectBounding;
		BOOL bDelete;

		double vecResult[3][3];//for subpixel
		int iMaxScoreIndex;//for subpixel
		BOOL bPosOnBorder;
		Point2d ptSubPixel;
		double dNewAngle;

		s_MatchParameter(Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
		{
			pt = ptMinMax;
			dMatchScore = dScore;
			dMatchAngle = dAngle;

			bDelete = FALSE;
			dNewAngle = 0.0;

			bPosOnBorder = FALSE;
		}
		s_MatchParameter()
		{
			double dMatchScore = 0;
			double dMatchAngle = 0;
		}
		~s_MatchParameter()
		{

		}
	};


	struct s_SingleTargetMatch
	{
		Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
		double dMatchedAngle;
		double dMatchScore;
	};
	
	
	struct s_BlockMax
	{
		struct Block
		{
			Rect rect;
			double dMax;
			Point ptMaxLoc;
			Block()
			{}
			Block(Rect rect_, double dMax_, Point ptMaxLoc_)
			{
				rect = rect_;
				dMax = dMax_;
				ptMaxLoc = ptMaxLoc_;
			}
		};
		s_BlockMax()
		{}
		vector<Block> vecBlock;
		Mat matSrc;
		s_BlockMax(Mat matSrc_, Size sizeTemplate)
		{
			matSrc = matSrc_;
			//將matSrc 拆成數個block，分別計算最大值
			int iBlockW = sizeTemplate.width * 2;
			int iBlockH = sizeTemplate.height * 2;

			int iCol = matSrc.cols / iBlockW;
			BOOL bHResidue = matSrc.cols % iBlockW != 0;

			int iRow = matSrc.rows / iBlockH;
			BOOL bVResidue = matSrc.rows % iBlockH != 0;

			if (iCol == 0 || iRow == 0)
			{
				vecBlock.clear();
				return;
			}

			vecBlock.resize(iCol * iRow);
			int iCount = 0;
			for (int y = 0; y < iRow; y++)
			{
				for (int x = 0; x < iCol; x++)
				{
					Rect rectBlock(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
					vecBlock[iCount].rect = rectBlock;
					minMaxLoc(matSrc(rectBlock), 0, &vecBlock[iCount].dMax, 0, &vecBlock[iCount].ptMaxLoc);
					vecBlock[iCount].ptMaxLoc += rectBlock.tl();
					iCount++;
				}
			}
			if (bHResidue && bVResidue)
			{
				Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
				Block blockRight;
				blockRight.rect = rectRight;
				minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
				blockRight.ptMaxLoc += rectRight.tl();
				vecBlock.push_back(blockRight);

				Rect rectBottom(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
				Block blockBottom;
				blockBottom.rect = rectBottom;
				minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
				blockBottom.ptMaxLoc += rectBottom.tl();
				vecBlock.push_back(blockBottom);
			}
			else if (bHResidue)
			{
				Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
				Block blockRight;
				blockRight.rect = rectRight;
				minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
				blockRight.ptMaxLoc += rectRight.tl();
				vecBlock.push_back(blockRight);
			}
			else
			{
				Rect rectBottom(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
				Block blockBottom;
				blockBottom.rect = rectBottom;
				minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
				blockBottom.ptMaxLoc += rectBottom.tl();
				vecBlock.push_back(blockBottom);
			}
		}
		void UpdateMax(Rect rectIgnore)
		{
			if (vecBlock.size() == 0)
				return;
			//找出所有跟rectIgnore交集的block
			int iSize = vecBlock.size();
			for (int i = 0; i < iSize; i++)
			{
				Rect rectIntersec = rectIgnore & vecBlock[i].rect;
				//無交集
				if (rectIntersec.width == 0 && rectIntersec.height == 0)
					continue;
				//有交集，更新極值和極值位置
				minMaxLoc(matSrc(vecBlock[i].rect), 0, &vecBlock[i].dMax, 0, &vecBlock[i].ptMaxLoc);
				vecBlock[i].ptMaxLoc += vecBlock[i].rect.tl();
			}
		}
		void GetMaxValueLoc(double& dMax, Point& ptMaxLoc)
		{
			int iSize = vecBlock.size();
			if (iSize == 0)
			{
				minMaxLoc(matSrc, 0, &dMax, 0, &ptMaxLoc);
				return;
			}
			//從block中找最大值
			int iIndex = 0;
			dMax = vecBlock[0].dMax;
			for (int i = 1; i < iSize; i++)
			{
				if (vecBlock[i].dMax >= dMax)
				{
					iIndex = i;
					dMax = vecBlock[i].dMax;
				}
			}
			ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
		}
	};


	class CMatchToolDlg
	{
	private:
		cv::Mat m_matSrc;
		cv::Mat m_matDst;
		s_TemplData m_TemplData;
		vector<s_SingleTargetMatch> m_vecSingleTargetData;


		void LearnPattern();
		int GetTopLayer(Mat* matTempl, int iMinDstLength);
		void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD);
		void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI);
		void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);
		Size  GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle);
		Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle);
		void FilterWithScore(vector<s_MatchParameter>* vec, double dScore);
		void FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod = CV_TM_CCOEFF_NORMED, double dMaxOverLap = 0);
		Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap);
		Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax);
		void SortPtWithCenter(vector<Point2f>& vecSort);
		BOOL SubPixEsimation(vector<s_MatchParameter>* vec, double* dX, double* dY, double* dAngle, double dAngleStep, int iMaxScoreIndex);
	
	
	public:
		/// <summary>
		int xt_m_ckSIMD = 1;
		int xt_m_iMinReduceArea = 256;    // 默认值
		double xt_m_dScore = 0.6;         // 匹配分数阈值
		int xt_m_iMaxPos = 10;
		int xt_m_bDebugMode = 0;

		int xt_m_bToleranceRange = 1;
		int xt_m_dTolerance1 = 0;
		int xt_m_dTolerance2 = 90;
		int xt_m_dTolerance3 = -180;
		int xt_m_dTolerance4 = -1;

		int xt_m_dToleranceAngle = 0;
		double xt_m_dMaxOverlap = 0.0;
		bool xt_m_bStopLayer1 = false;
		/// </summary>

		CMatchToolDlg();
		BOOL Match(cv::Mat& src, cv::Mat& dst);

	};




    void A119_solver();
}


#endif 
