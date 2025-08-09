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


/***
计算出溢胶区域的面积
***/
void A101();


/***
将一个不规则图像的轮廓使用findcontour以外的方法呈现(morphologyEx)
***/
void A102();



/***
将一个不规则图像的轮廓使用findcontour以外的方法呈现(morphologyEx)
***/
void A103();



/***
基于角度将图像进行透视变换
***/
void A104();


/***
亚像素
***/
void A105();



/***
测试blob
***/
namespace NA106 {
	void A106();
}



/***
可以被重写
***/
namespace NA107 {
	void A107();
}



/***
可以被重写
***/
namespace NA108 {
	void A108_solver();
}



/*
功能：测试omp和parellelfor方式的加速效果
*/
namespace NA109 {
	void A109_solver();
}


/*
功能：通过SSE\多路并行计算等方式优化RGB2Y
*/
namespace NA110 {
	void A110_solver();
}




/*
功能：通过SSE\多路并行计算等方式优化picshadowx
*/
namespace NA111 {
	int picshadowx(cv::Mat binary, cv::Mat* show);
	void doing(cv::Mat imgori);
	void A111_solver();
}



/*
功能：深度图与点云图互转(获取实际像素点的横坐标)
*/
namespace NA112 {
	void A112_solver();
}




/***
测试GX项目中各个功能函数的批量执行效率；
Case1:验证功能：多张图构建成一张图后再进行缺陷检测
Case2:一张图像，先行后列遍历和先列后行遍历的执行效率差别
Case3: 两张图像横向拼接在一起
Case5: 图像如果是竖直方向的，则执行竖直方向的投影检测的执行效率情况
Case6: 研究均值滤波和SSE
Case7: 测试各版本投影操作的执行效率
Case8: 测试中心区域增强的执行效率
***/
namespace NA113 {
	void A113_solver();
}




/***
C++开发toy functions;
Case1: 获取memory信息；
***/
namespace NA114 {
	void A114_solver();
}





/***
* @brief A115_solver
* function: C++中Opencv的传统机器学习算法
Case1: SVM+GX
***/
namespace NA115 {
    void A115_solver();
}




/***
* @brief A116_solver
* function: 学习C++的知识：
Case1: C++的控制赋值
***/
namespace NA116 {
    void A116_solver();
}




/***
* @brief A117_solver
* function: 线程相关知识;
Case1: 基于线程模仿算子计算队列和图像保存队列的处理效率随外部系统应用运行的变化；
***/
namespace NA117 {
    void A117_solver();
}





/***
* @brief A118_solver
* function: 图像处理的常用方法和技巧(不断积累)
* 
传统图像处理常用方法：
1. （辅助）颜色空间转换：提取某个通道色彩或者增强某个物体的对比度
2. 形态学操作：去除噪声、提取边界（黑帽）、连接和分割对象；
3. （辅助）图像增强：灰度变换、直方图均衡、锐化等方法，增强目标物体间的对比度
4. 图像分割和特征提取：阈值分割、轮廓提取、分水岭算法（*），提升目标区域的对比度，将目标区域从图像中分割出来
5. （辅助）图像去噪：利用中值滤波去除椒盐噪声、通过傅里叶变换进行频域处理；
6. （优化）图像格式转换与优化：指针数据与Mat类型变量的转换
7. 特征点检测与匹配：（非工业视觉场景）


* Case1: 基于连通域信息能够提取出图像中的目标特征点
*
* Case3: 二八原则的均值计算技巧
* 去除噪声数据，提升估计值的重复计算的稳定性以及准确度！
*
* 场景：
* 1. 根据3D图像中铆钉有弧形的上表面深度值的集合，估计搞铆钉的高度
* 2. 根据鲜花整个茎干的直径的集合，估计茎干的合理直径值
* 
* Case4: 顶帽和黑帽的应用场景:
* 1. 顶帽：本质上一句话，为了分离比邻近点亮一些的斑块；（原图 - 原图的开运算结果）
* 2. 黑帽：本质上一句话，为了分离比邻近点暗一些的斑块；（原图的闭运算结果 - 原图）
* Case5:
* 1. 顶帽操作中不同开运算操作和滤波操作对黑斑/亮斑提取效果的实验：抵抗产品图像亮度不均匀的问题
* 2. 不同颜色的产品如何统一处理获取产品轮廓
* 3. 自适应直方图
***/
namespace NA118 {
    void A118_solver();
}







/***
* * @brief A119_solver
* 算子参考源码：https://github.com/DennisLiu1993/Fastest_Image_Pattern_Matching
* 算子功能：快速图像模板匹配，多角度、多尺度匹配
*
* Demo:
* std::string src_path = "H:\\Projects\\Fastest_Image_Pattern_Matching-main\\Test Images\\Src1.bmp";
* std::string dst_path = "H:\\Projects\\Fastest_Image_Pattern_Matching-main\\Test Images\\20220611.bmp";
* NCCMatchConfig config;
* cv::Mat xt_m_matSrc = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
* cv::Mat xt_m_matDst = cv::imread(dst_path, cv::IMREAD_GRAYSCALE);
* CMatchToolDlg matcher;
* matcher.SetConfig(config);
* matcher.Match(xt_m_matSrc, xt_m_matDst);
**/
namespace NA119 {

	const Scalar colorWaterBlue(230, 255, 102);
	const Scalar colorBlue(255, 0, 0);
	const Scalar colorYellow(0, 255, 255);
	const Scalar colorRed(0, 0, 255);
	const Scalar colorBlack(0, 0, 0);
	const Scalar colorGray(200, 200, 200);
	const Scalar colorSystem(240, 240, 240);
	const Scalar colorGreen(0, 255, 0);
	const Scalar colorWhite(255, 255, 255);
	const Scalar colorPurple(214, 112, 218);
	const Scalar colorGoldenrod(15, 185, 255);



	//!? 存储模板匹配的相关数据和状态，可能是用于多尺度金字塔匹配或者某种优化的模板匹配算法比如归一化互相关。
	struct s_TemplData
	{
		vector<cv::Mat> vecPyramid;         // 存储不同尺度的模板图像金字塔
		vector<cv::Scalar> vecTemplMean;    // 每个金字塔层模板的均值（BGR或者灰度）
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
		cv::Point2d pt;
		double dMatchScore;
		double dMatchAngle;
		//Mat matRotatedSrc;
		Rect rectRoi;
		double dAngleStart;
		double dAngleEnd;
		cv::RotatedRect rectR;
		cv::Rect rectBounding;
		BOOL bDelete;

		double vecResult[3][3];//for subpixel
		int iMaxScoreIndex;//for subpixel
		BOOL bPosOnBorder;
		cv::Point2d ptSubPixel;
		double dNewAngle;

		//, Mat matRotatedSrc = Mat ())
		s_MatchParameter(cv::Point2f ptMinMax, double dScore, double dAngle)
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
		cv::Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
		double dMatchedAngle;
		double dMatchScore;
	};
	
	
	struct s_BlockMax
	{
		struct Block
		{
			cv::Rect rect;
			double dMax;
			cv::Point ptMaxLoc;
			Block()
			{}
			Block(cv::Rect rect_, double dMax_, cv::Point ptMaxLoc_)
			{
				rect = rect_;
				dMax = dMax_;
				ptMaxLoc = ptMaxLoc_;
			}
		};
		s_BlockMax()
		{}
		vector<Block> vecBlock;
		cv::Mat matSrc;
		s_BlockMax(cv::Mat matSrc_, cv::Size sizeTemplate)
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
		void UpdateMax(cv::Rect rectIgnore)
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
		void GetMaxValueLoc(double& dMax, cv::Point& ptMaxLoc)
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

	struct NCCMatchConfig
	{
		int useSIMD = 1;                // 是否使用SIMD加速
		int minReduceArea = 256;        // 最小金字塔区域
		double scoreThreshold = 0.6;    // 匹配分数阈值
		int maxMatchCount = 10;         // 最大返回匹配数
		int debugMode = 0;              // 是否调试模式
		int useToleranceRange = 1;      // 是否使用角度范围
		double tolerance1 = 0;          // 角度范围1起点
		double tolerance2 = 90;         // 角度范围1终点
		double tolerance3 = -180;       // 角度范围2起点
		double tolerance4 = -1;         // 角度范围2终点
		double toleranceAngle = 0;      // 对称角度范围
		double maxOverlap = 0.0;        // 最大重叠比例
		bool stopAtLayer1 = false;      // 是否只到第一层
	};




	void DrawMatchResults(const cv::Mat& srcImage, const std::vector<s_SingleTargetMatch>& matches, const std::string& outputPath = "result.jpg");
	inline bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs);
	inline bool comparePtWithAngle(const pair<Point2f, double> lhs, const pair<Point2f, double> rhs);
	inline bool compareMatchResultByPos(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs);
	inline bool compareMatchResultByScore(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs);
	inline bool compareMatchResultByPosX(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs);
	inline int _mm_hsum_epi32(__m128i V);
	inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char* pCharConv, int iLength);


	class CMatchToolDlg
	{
	private:
		cv::Mat m_matSrc;
		cv::Mat m_matDst;
		s_TemplData m_TemplData;
		vector<s_SingleTargetMatch> m_vecSingleTargetData;





		void LearnPattern();
		
		int GetTopLayer(cv::Mat* matTempl, int iMinDstLength);
		
		void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, BOOL bUseSIMD);
		
		void GetRotatedROI(cv::Mat& matSrc, cv::Size size, cv::Point2f ptLT, double dAngle, cv::Mat& matROI);
		
		void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);
		
		cv::Size GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, double dRAngle);
		
		cv::Point2f ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, double dAngle);
		
		void FilterWithScore(vector<s_MatchParameter>* vec, double dScore);
		
		void FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod = CV_TM_CCOEFF_NORMED, double dMaxOverLap = 0);
		
		cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap);
		
		cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax);
		
		void SortPtWithCenter(vector<cv::Point2f>& vecSort);
		
		BOOL SubPixEsimation(vector<s_MatchParameter>* vec, double* dX, double* dY, double* dAngle, double dAngleStep, int iMaxScoreIndex);
	
	
	public:
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


		// 构造函数
		CMatchToolDlg();
		
		// 执行匹配
		BOOL Match(cv::Mat& src, cv::Mat& dst);
		
		// 读取配置参数
		void SetConfig(const NCCMatchConfig& config);
	};



    void A119_solver();
}


#endif 
