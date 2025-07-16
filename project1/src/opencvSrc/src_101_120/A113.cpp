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
#include <algorithm>

using namespace std;
using namespace cv;

/*
功能：单元测试三行代码算法的效率

验证：
1. 一张图像，先行后列遍历和先列后行遍历的差别；
2. 一张原始图像的数据存储是连续的；但是如果反转后如何保证数据存储也是连续的？
3. 

*/

#define Gaussian_Size 20
#define Gaussian_Size_2 (Gaussian_Size>>1)

#pragma execution_character_set("utf-8") 

namespace NA113 {
	int NUMTHREADS = 4;

	// 参数：将认定为缺陷的区域和认定为非缺陷的区域的差异更加明显化！
	double FEATURE1_CONTRAST_DEGREE = 45;    // 50      值越小，缺陷越多
	double DEFECT_LENGTH = 8.5;    // 5.3()   不能实现
	double DEFECT_LENGTH2 = 5.3;   // 5.3()   
	double DEFECT1_WIDTH = 3;


	double FEATURE2_CONTRAST_DEGREE = 45;
	double FEATURE2_BLOCK_LENGTH = 3;
	double FEATURE2_BLOCK_NUMS = 3;
	double DEFECT2_WIDTH = 3;


	double FEATURE3_CONTRAST_DEGREE = 20;
	double FEATURE3_DIAMETER = 44;
	double FEATURE3_DIAMETER_THLD = 3;
	double DEFECT3_WIDTH = 3;


	int BLUR_RADIUS = 10;
	// 250μm （50, 5。3）；   125μm（50， 3.3）


	// 绘制
	int DRAW_MAX_H = 15;
	int NG_DRAW_MAX_H = 5;


	float Gaussian_Ker_XY[Gaussian_Size];

	class BlurVersion2
	{
	public:
		int IM_GetMirrorPos(int Length, int Pos)
		{
			if (Pos < 0)
				return -Pos;
			else if (Pos >= Length)
				return Length + Length - Pos - 2;
			else
				return Pos;
		}

		void FillLeftAndRight_Mirror_C(int* Array, int Length, int Radius)
		{
			for (int X = 0; X < Radius; X++)
			{
				Array[X] = Array[Radius + Radius - X];
				Array[Radius + Length + X] = Array[Radius + Length - X - 2];
			}

		}

		void FillLeftAndRight_Mirror_SSE(int* Array, int Length, int Radius)
		{
			int BlockSize = 4, Block = Radius / BlockSize;
			int X = 0;
			for (; X + BlockSize < Radius; X += BlockSize)
			{
				__m128i SrcV1 = _mm_loadu_si128((__m128i*)(Array + Radius + Radius - X - 3));
				__m128i SrcV2 = _mm_loadu_si128((__m128i*)(Array + Radius + Length - X - 5));
				_mm_storeu_si128((__m128i*)(Array + X), _mm_shuffle_epi32(SrcV1, _MM_SHUFFLE(0, 1, 2, 3)));
				_mm_storeu_si128((__m128i*)(Array + Radius + Length + X), _mm_shuffle_epi32(SrcV2, _MM_SHUFFLE(0, 1, 2, 3)));
			}
			for (; X < Radius; X++)
			{
				Array[X] = Array[Radius + Radius - X];
				Array[Radius + Length + X] = Array[Radius + Length - X - 2];
			}
		}

		// 计算一个数组的所有的元素的和
		int SumofArray_C(int* Array, int Length)
		{
			int Sum = 0;
			for (int X = 0; X < Length; X++)
			{
				Sum += Array[X];
			}
			return Sum;
		}

		void _mm_storesi128_4char(__m128i Src, unsigned char* Dest)
		{
			__m128i T = _mm_packs_epi32(Src, Src);
			T = _mm_packus_epi16(T, T);
			*((int*)Dest) = _mm_cvtsi128_si32(T);
		}

		int SumofArray_SSE(int* Array, int Length)
		{
			int BlockSize = 8, Block = Length / BlockSize;
			__m128i Sum1 = _mm_setzero_si128();
			__m128i Sum2 = _mm_setzero_si128();

			int X = 0;
			for (; X + BlockSize < Length; X += BlockSize)
			{
				Sum1 = _mm_add_epi32(Sum1, _mm_loadu_si128((__m128i*)(Array + X + 0)));
				Sum2 = _mm_add_epi32(Sum2, _mm_loadu_si128((__m128i*)(Array + X + 4)));
			}
			// 水平相加Sum1和Sum2的32位整型
			__m128i SumTotal = _mm_add_epi32(Sum1, Sum2);   // d c b a

			// 将128位寄存器拆分为高位和低位相加
			__m128i shuf = _mm_shuffle_epi32(SumTotal, _MM_SHUFFLE(2, 3, 0, 1)); // [b,a,d,c]
			__m128i sums = _mm_add_epi32(SumTotal, shuf);                        // [d+b, c+a, b+d, a+c] 
			__m128i sumt = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));     // [c+a, d+b, a+c, b+d]  
			int Sum = _mm_cvtsi128_si32(_mm_add_epi32(sums, sumt));              // 
			/**
			* SumTotal: | v3 | v2 | v1 | v0 |  // v0在内存低地址，v3在高地址
			* __m128i shuf = _mm_shuffle_epi32(SumTotal, _MM_SHUFFLE(2, 3, 0, 1): shuf: | v1 | v0 | v3 | v2 | 
			* _mm_cvtsi128_si32 提取这个 128 位向量的低 32 位，并将其作为一个 int 返回
			*/


			//　　处理剩余不能被SSE优化的数据
			for (; X < Length; X++)
			{
				Sum += Array[X];
			}
			return Sum;
		}

		inline unsigned char IM_ClampToByte(int Value)            //    现代PC还是这样直接写快些
		{
			if (Value < 0)
				return 0;
			else if (Value > 255)
				return 255;
			else
				return (unsigned char)Value;
			//return ((Value | ((signed int)(255 - Value) >> 31)) & ~((signed int)Value >> 31));
		}

		// ori
		int IM_BoxBlur_C(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int Radius)
		{
			// 参数有效性检查
			int Channel = Stride / Width;                                            // 计算通道数
			if ((Src == NULL) || (Dest == NULL))                       return 0;     // 空指针检查
			if ((Width <= 0) || (Height <= 0) || (Radius <= 0))        return 0;     // 尺寸有效性检查
			if ((Channel != 1) && (Channel != 3) && (Channel != 4))    return 0;     // 通道数检查


			Radius = (std::min)((std::min)(Radius, Width - 1), Height - 1);          // 限制最大半径
			int SampleAmount = (2 * Radius + 1) * (2 * Radius + 1);                  // 总采样像素数
			float Inv = 1.0 / SampleAmount;                                          // 用于求平均的倒数


			// 内存分配
			int* ColValue = (int*)malloc((Width + Radius + Radius) * (Channel == 1 ? Channel : 4) * sizeof(int));  // 列累加值缓存
			int* ColOffset = (int*)malloc((Height + Radius + Radius) * sizeof(int));                               // 行镜像索引表
			if ((ColValue == NULL) || (ColOffset == NULL))
			{
				if (ColValue != NULL)    free(ColValue);
				if (ColOffset != NULL)    free(ColOffset);
				return 0;
			}

			// 创建行镜像索引表（处理图像边界）
			for (int Y = 0; Y < Height + Radius + Radius; Y++)
				ColOffset[Y] = IM_GetMirrorPos(Height, Y - Radius);  // 生成镜像行坐标

			// 单通道处理
			if (Channel == 1)
			{
				for (int Y = 0; Y < Height; Y++)                                                  // 遍历每一行
				{
					// 列缓存初始化/更新
					unsigned char* LinePD = Dest + Y * Stride;
					if (Y == 0)                                                                   // 首行初始化
					{
						memset(ColValue + Radius, 0, Width * sizeof(int));
						for (int Z = -Radius; Z <= Radius; Z++)                                   // 累加初始列和
						{
							unsigned char* LinePS = Src + ColOffset[Z + Radius] * Stride;         // 获取镜像行
							for (int X = 0; X < Width; X++)                                       // 逐列累加
							{
								ColValue[X + Radius] += LinePS[X];
							}
						}
					}
					else                                                                            // 后续行快速更新
					{
						unsigned char* RowMoveOut = Src + ColOffset[Y - 1] * Stride;                // 移出行的指针
						unsigned char* RowMoveIn = Src + ColOffset[Y + Radius + Radius] * Stride;   // 移入行的指针
						for (int X = 0; X < Width; X++)                                             // 列和增量更新
						{
							ColValue[X + Radius] -= RowMoveOut[X] - RowMoveIn[X];
						}
					}

					// 边缘镜像处理
					FillLeftAndRight_Mirror_C(ColValue, Width, Radius);                  // 列缓存镜像填充

					// 横向滑动窗口求和
					int LastSum = SumofArray_C(ColValue, Radius * 2 + 1);                // 初始窗口和
					LinePD[0] = IM_ClampToByte(LastSum * Inv);                           // 计算首像素
					for (int X = 1; X < Width; X++)                                                // 滑动窗口优化
					{
						int NewSum = LastSum - ColValue[X - 1] + ColValue[X + Radius + Radius];    // 增量更新窗口和
						LinePD[X] = IM_ClampToByte(NewSum * Inv);                                  // 计算结果像素
						LastSum = NewSum;
					}
				}
			}
			else if (Channel == 3)
			{

			}
			else if (Channel == 4)
			{

			}
			free(ColValue);  // 释放内存
			free(ColOffset);
			return 1;
		}

		// comment版本
		int IM_BoxBlur_C_Comment(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int Radius)
		{
			/** 参数有效性检查 */
			int Channel = Stride / Width;                                            // 计算通道数
			if ((Src == NULL) || (Dest == NULL))                       return 0;     // 空指针检查
			if ((Width <= 0) || (Height <= 0) || (Radius <= 0))        return 0;     // 尺寸有效性检查
			if ((Channel != 1) && (Channel != 3) && (Channel != 4))    return 0;     // 通道数检查


			Radius = (std::min)((std::min)(Radius, Width - 1), Height - 1);          // 限制最大半径
			int SampleAmount = (2 * Radius + 1) * (2 * Radius + 1);                  // 总采样像素数
			float Inv = 1.0 / SampleAmount;                                          // 用于求平均的倒数


			/** 内存分配 */ 
			int* ColValue = (int*)malloc((Width + Radius + Radius) * (Channel == 1 ? Channel : 4) * sizeof(int));  // 列累加值缓存
			int* ColOffset = (int*)malloc((Height + Radius + Radius) * sizeof(int));                               // 行镜像索引表
			if ((ColValue == NULL) || (ColOffset == NULL))
			{
				if (ColValue != NULL)     free(ColValue);
				if (ColOffset != NULL)    free(ColOffset);
				return 0;
			}

			/** 创建行镜像索引表（处理图像边界），创建后不会被修改！ */ 
			for (int Y = 0; Y < Height + Radius + Radius; Y++)
				// pos 的取值范围是 [-Radius, Height + Radius]
				ColOffset[Y] = IM_GetMirrorPos(Height, Y - Radius);  // 生成镜像行坐标

			//! comment
			//for (int i = 0; i < Height + Radius + Radius; i++) {
			//	LOGD("ColOffset[{}] = {};", i, ColOffset[i]);
			//}
			// 1 0123 2


			// 单通道处理
			if (Channel == 1)
			{
				for (int Y = 0; Y < Height; Y++)                                                  // 遍历每一行
				{
					/** 获取Dst的某一行的位置索引 */
					unsigned char* LinePD = Dest + Y * Stride;                                    
					if (Y == 0)                                                                   // 首行初始化
					{
						/** 初始化ColValue的值为0和-842150451 */
						memset(ColValue + Radius, 0, Width * sizeof(int));
						//for (int i = 0; i < Width + Radius + Radius; i++) {
						//	LOGD("ColValue[{}] = {};", i, ColValue[i]);
						//}
						//  -842150451 | 0 0 0 0 0 | -842150451 
						 
						
						// ColValue 空间大小为 Width + Radius + Radius， 现在从 Radius开始到Radius+width之间的位置设置成 像素0
						// radius长度的未赋值区域 + width长度的0赋值区域 + radius长度的未赋值区域
						for (int Z = -Radius; Z <= Radius; Z++)                                   // 累加初始列和
						{
							unsigned char* LinePS = Src + ColOffset[Z + Radius] * Stride;         //! 获取镜像行(虚的和实的)的起始位置
							for (int X = 0; X < Width; X++)                                       // 逐列累加镜像行的元素值 赋值给
							{
								ColValue[X + Radius] += LinePS[X]; 
							}

							//for (int i = 0; i < Width + Radius + Radius; i++) {
							//	LOGD("ColValue[{}] = {};", i, ColValue[i]);
							//}
							// -842150451 | 6  7  8  9  10 | -842150451 
							// -842150451 | 1  2  3  4  5 | -842150451 
							// -842150451 | 6  7  8  9  10 | -842150451 

						}

						//for (int i = 0; i < Width + Radius + Radius; i++) {
						//	LOGD("ColValue[{}] = {};", i, ColValue[i]);
						//}
						// -842150451 | 13 16 19 22 25 | -842150451 

					}
					else                                                                            // 后续行快速更新
					{
						unsigned char* RowMoveOut = Src + ColOffset[Y - 1] * Stride;                // 移出行的指针
						unsigned char* RowMoveIn = Src + ColOffset[Y + Radius + Radius] * Stride;   // 移入行的指针
						for (int X = 0; X < Width; X++)                                             // 增量更新
						{
							ColValue[X + Radius] -= RowMoveOut[X] - RowMoveIn[X];
						}
						// Y = 1:
						// ColValue:   16 | 13 16 19 22 25 | 22
						// RowMoveOut: 7  | 6  7  8  9  10 | 9
						// RowMoveIn:  12 | 11 12 13 14 15 | 14
						// 增量:       5  |  5  5  5  5  5 |  5
					}

					/** 边缘镜像处理 */

					FillLeftAndRight_Mirror_C(ColValue, Width, Radius);                  // 列缓存镜像填充
					//for (int i = 0; i < Width + Radius + Radius; i++) {
					//	LOGD("ColValue[{}] = {};", i, ColValue[i]);
					//}
					// 将 -842150451 | 13 16 19 22 25 | -842150451 中的 -842150451 赋值为其对应的镜像值
					// 即 16 | 13 16 19 22 25 | 22 

					/** 横向滑动窗口求和 */ 
					int LastSum = SumofArray_C(ColValue, Radius * 2 + 1);                // 初始窗口和
					LinePD[0] = IM_ClampToByte(LastSum * Inv);                           // 控制和的范围(0~255边界和整数化)并赋值给dst
					for (int X = 1; X < Width; X++)                                                // 滑动窗口优化
					{
						int NewSum = LastSum - ColValue[X - 1] + ColValue[X + Radius + Radius];    // 增量更新窗口和: 基于首个和，减去前一个，加上后一个
						LinePD[X] = IM_ClampToByte(NewSum * Inv);                                  // 控制和的范围(0~255边界和整数化)并赋值给dst
						LastSum = NewSum;
					}
				}
			}
			else if (Channel == 3)
			{

			}
			else if (Channel == 4)
			{

			}
			free(ColValue);  // 释放内存
			free(ColOffset);
			return 1;
		}

		int IM_BoxBlur_SSE(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int Radius)
		{
			int Channel = Stride / Width;
			if ((Src == NULL) || (Dest == NULL))
				return 0;
			if ((Width <= 0) || (Height <= 0) || (Radius <= 0))
				return 0;
			if ((Channel != 1) && (Channel != 3) && (Channel != 4))
				return 0;


			Radius = (std::min)((std::min)(Radius, Width - 1), Height - 1);        //    由于镜像的需求，要求半径不能大于宽度或高度-1的数据
			int SampleAmount = (2 * Radius + 1) * (2 * Radius + 1);
			float Inv = 1.0 / SampleAmount;


			int* ColValue = (int*)malloc((Width + Radius + Radius) * (Channel == 1 ? Channel : 4) * sizeof(int));
			int* ColOffset = (int*)malloc((Height + Radius + Radius) * sizeof(int));


			if ((ColValue == NULL) || (ColOffset == NULL))
			{
				if (ColValue != NULL)    free(ColValue);
				if (ColOffset != NULL)    free(ColOffset);
				return 0;
			}

			for (int Y = 0; Y < Height + Radius + Radius; Y++)
				ColOffset[Y] = IM_GetMirrorPos(Height, Y - Radius);

			if (Channel == 1)
			{
				for (int Y = 0; Y < Height; Y++)
				{
					unsigned char* LinePD = Dest + Y * Stride;
					if (Y == 0)
					{
						memset(ColValue + Radius, 0, Width * sizeof(int));
						for (int Z = -Radius; Z <= Radius; Z++)
						{
							unsigned char* LinePS = Src + ColOffset[Z + Radius] * Stride;

							int BlockSize = 8, Block = Width / BlockSize;
							int X = 0;
							for (; X < Width; X += BlockSize)
							{
								int* DestP = ColValue + X + Radius;
								__m128i Sample = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(LinePS + X)));
								_mm_storeu_si128((__m128i*)DestP, _mm_add_epi32(_mm_loadu_si128((__m128i*)DestP), _mm_cvtepi16_epi32(Sample)));
								_mm_storeu_si128((__m128i*)(DestP + 4), _mm_add_epi32(_mm_loadu_si128((__m128i*)(DestP + 4)), _mm_unpackhi_epi16(Sample, _mm_setzero_si128())));
							}

							for (; X < Width; X++)
							{
								ColValue[X + Radius] += LinePS[X];                                            //    更新列数据
							}
						}
					}
					else
					{
						unsigned char* RowMoveOut = Src + ColOffset[Y - 1] * Stride;                //    即将减去的那一行的首地址
						unsigned char* RowMoveIn = Src + ColOffset[Y + Radius + Radius] * Stride;    //    即将加上的那一行的首地址

						int BlockSize = 8, Block = Width / BlockSize;
						__m128i Zero = _mm_setzero_si128();
						int X = 0;
						for (; X < Width; X += BlockSize)
						{
							int* DestP = ColValue + X + Radius;
							__m128i MoveOut = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(RowMoveOut + X)), Zero);
							__m128i MoveIn = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(RowMoveIn + X)), Zero);
							__m128i Diff = _mm_sub_epi16(MoveIn, MoveOut);                        //    注意这个有负数也有正数的，有负数时转换为32位是不能用_mm_unpackxx_epi16体系的函数
							_mm_storeu_si128((__m128i*)DestP, _mm_add_epi32(_mm_loadu_si128((__m128i*)DestP), _mm_cvtepi16_epi32(Diff)));
							_mm_storeu_si128((__m128i*)(DestP + 4), _mm_add_epi32(_mm_loadu_si128((__m128i*)(DestP + 4)), _mm_cvtepi16_epi32(_mm_srli_si128(Diff, 8))));
						}
						for (; X < Width; X++)
						{
							ColValue[X + Radius] -= RowMoveOut[X] - RowMoveIn[X];                                            //    更新列数据
						}
					}
					FillLeftAndRight_Mirror_SSE(ColValue, Width, Radius);                  //    镜像填充左右数据
					int LastSum = SumofArray_C(ColValue, Radius * 2 + 1);                  //    处理每行第一个数据
					LinePD[0] = IM_ClampToByte(LastSum * Inv);

					int BlockSize = 4, Block = (Width - 1) / BlockSize;
					__m128i OldSum = _mm_set1_epi32(LastSum);
					__m128 Inv128 = _mm_set1_ps(Inv);

					int X = 1;

					for (; X < Width; X += BlockSize)
					{
						__m128i ColValueOut = _mm_loadu_si128((__m128i*)(ColValue + X - 1));
						__m128i ColValueIn = _mm_loadu_si128((__m128i*)(ColValue + X + Radius + Radius));
						__m128i ColValueDiff = _mm_sub_epi32(ColValueIn, ColValueOut);                            //    P3 P2 P1 P0                                                
						__m128i Value_Temp = _mm_add_epi32(ColValueDiff, _mm_slli_si128(ColValueDiff, 4));        //    P3+P2 P2+P1 P1+P0 P0
						__m128i Value = _mm_add_epi32(Value_Temp, _mm_slli_si128(Value_Temp, 8));                 //    P3+P2+P1+P0 P2+P1+P0 P1+P0 P0
						__m128i NewSum = _mm_add_epi32(OldSum, Value);
						OldSum = _mm_shuffle_epi32(NewSum, _MM_SHUFFLE(3, 3, 3, 3));                              //    重新赋值为最新值
						__m128 Mean = _mm_mul_ps(_mm_cvtepi32_ps(NewSum), Inv128);
						_mm_storesi128_4char(_mm_cvtps_epi32(Mean), LinePD + X);
					}

					for (; X < Width; X++)
					{
						int NewSum = LastSum - ColValue[X - 1] + ColValue[X + Radius + Radius];
						LinePD[X] = IM_ClampToByte(NewSum * Inv);
						LastSum = NewSum;
					}
				}
			}
			else if (Channel == 3)
			{

			}
			else if (Channel == 4)
			{

			}
			free(ColValue);
			free(ColOffset);
			return 1;
		}

		int IM_BoxBlur_SSE_Comment(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int Radius)
		{
			int Channel = Stride / Width;
			if ((Src == NULL) || (Dest == NULL))
				return 0;
			if ((Width <= 0) || (Height <= 0) || (Radius <= 0))
				return 0;
			if ((Channel != 1) && (Channel != 3) && (Channel != 4))
				return 0;


			Radius = (std::min)((std::min)(Radius, Width - 1), Height - 1);        //    由于镜像的需求，要求半径不能大于宽度或高度-1的数据
			int SampleAmount = (2 * Radius + 1) * (2 * Radius + 1);
			float Inv = 1.0 / SampleAmount;


			int* ColValue = (int*)malloc((Width + Radius + Radius) * (Channel == 1 ? Channel : 4) * sizeof(int));
			int* ColOffset = (int*)malloc((Height + Radius + Radius) * sizeof(int));


			if ((ColValue == NULL) || (ColOffset == NULL))
			{
				if (ColValue != NULL)    free(ColValue);
				if (ColOffset != NULL)    free(ColOffset);
				return 0;
			}

			for (int Y = 0; Y < Height + Radius + Radius; Y++)
				ColOffset[Y] = IM_GetMirrorPos(Height, Y - Radius);

			if (Channel == 1)
			{
				for (int Y = 0; Y < Height; Y++)
				{
					unsigned char* LinePD = Dest + Y * Stride;
					if (Y == 0)
					{
						memset(ColValue + Radius, 0, Width * sizeof(int));
						for (int Z = -Radius; Z <= Radius; Z++)
						{
							unsigned char* LinePS = Src + ColOffset[Z + Radius] * Stride;

							int BlockSize = 8, Block = Width / BlockSize;
							int X = 0;
							for (; X + BlockSize < Width; X += BlockSize)
							{
								int* DestP = ColValue + X + Radius;
								__m128i Sample = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(LinePS + X)));
								_mm_storeu_si128((__m128i*)DestP, _mm_add_epi32(_mm_loadu_si128((__m128i*)DestP), _mm_cvtepi16_epi32(Sample)));
								_mm_storeu_si128((__m128i*)(DestP + 4), _mm_add_epi32(_mm_loadu_si128((__m128i*)(DestP + 4)), _mm_unpackhi_epi16(Sample, _mm_setzero_si128())));
								/**
								* _mm_cvtepu8_epi16: 指令会将这16个8位数中的低8个零扩展到16位，成为8个16位整数，填充到输出寄存器中。因为128位除以16位是8个元素，所以高8个8位可能被忽略。
								* _mm_unpackhi_epi16: 将高64位数据移动到低64位位置
								* _mm_unpackhi_epi16(a, b): 
								**** a = [a7, a6, a5, a4, a3, a2, a1, a0]  // 高64位：a7, a6, a5, a4
								**** b = [b7, b6, b5, b4, b3, b2, b1, b0]  // 高64位：b7, b6, b5, b4
								**** 结果 = [b7, a7, b6, a6, b5, a5, b4, a4] 
								* _mm_loadu_si128((__m128i*)(DestP + 4)) 得到4个32位整数；_mm_unpackhi_epi16(Sample, _mm_setzero_si128()) 得到8个16位整数，而当_mm_add_epi32处理时会将后者视为四个32位整数进行分段处理，而分段使得b7a7成为一个整体。以达到Sample高4位的计算。
								*/
							}

							for (; X < Width; X++)
							{
								ColValue[X + Radius] += LinePS[X];                                            //    更新列数据
							}
						}
					}
					else
					{
						unsigned char* RowMoveOut = Src + ColOffset[Y - 1] * Stride;                //    即将减去的那一行的首地址
						unsigned char* RowMoveIn = Src + ColOffset[Y + Radius + Radius] * Stride;    //    即将加上的那一行的首地址

						int BlockSize = 8, Block = Width / BlockSize;
						__m128i Zero = _mm_setzero_si128(); 
						int X = 0;
						for (; X + BlockSize < Width; X += BlockSize)
						{
							int* DestP = ColValue + X + Radius;
							__m128i MoveOut = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(RowMoveOut + X)), Zero);
							__m128i MoveIn = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(RowMoveIn + X)), Zero);
							__m128i Diff = _mm_sub_epi16(MoveIn, MoveOut);   
							// 注意这个有负数也有正数的，有负数时转换为32位是不能用_mm_unpackxx_epi16体系的函数
							_mm_storeu_si128((__m128i*)DestP, _mm_add_epi32(_mm_loadu_si128((__m128i*)DestP), _mm_cvtepi16_epi32(Diff)));
							_mm_storeu_si128((__m128i*)(DestP + 4), _mm_add_epi32(_mm_loadu_si128((__m128i*)(DestP + 4)), _mm_cvtepi16_epi32(_mm_srli_si128(Diff, 8))));
							/**
							* _mm_setzero_si128: si128表示scalar操作 + 有符号128位整数。
							* _mm_loadl_epi64: 从内存加载64位整数到寄存器的低64位（高64位清零） 使用场景：加载8个8位整数或4个16位整数时使用
							* (__m128i*)(RowMoveOut + X): 将 RowMoveOut + X 的地址转换为一个指向 __m128i 类型的指针
							* __m128i 是 SSE 指令集中的数据类型，它表示一个 128 位的整数向量，可以包含 16 个字节的数据
							* _mm_unpacklo_epi8 是SSE2指令集中的数据重组指令，用于将两个128位寄存器的低64位数据（前8个8位整数）进行交错排列
							**** __m128i _mm_unpacklo_epi8(__m128i a, __m128i b):
							**** b = [b15,b14,...,b1,b0]        // 寄存器b的低64位：b7,b6...b0
							**** a = [a15,a14,...,a1,a0]        // 寄存器a的低64位：a7,a6...a0
							**** 结果 = [b7a7,b6a6,...,b0a0] // 交错排列后的128位数据
							* epi8，即8位有/无符号整数
							**** // 操作示意图（每个短整型为16位）：
							**** a = [a7, a6, a5, a4, a3, a2, a1, a0]
							**** b = [b7, b6, b5, b4, b3, b2, b1, b0]
							**** 结果 = [a7-b7, a6-b6, ..., a0-b0]
							* ——————————————————————————————————
							* _mm_loadu_si128: u表示不需要对齐加载，si表示signed integer
							* _mm_cvtepi16_epi32：只处理输入寄存器的低64位（前4个16位元素）
							* _mm_srli_si128：逻辑右移（shift right logical，从高地址向低地址的方向！） 移动8个字节，也就是64位；
							
							*/
						}
						for (; X < Width; X++)
						{
							ColValue[X + Radius] -= RowMoveOut[X] - RowMoveIn[X];                                            //    更新列数据
						}
					}
					FillLeftAndRight_Mirror_SSE(ColValue, Width, Radius);                  //    镜像填充左右数据
					int LastSum2 = SumofArray_C(ColValue, Radius * 2 + 1);                  //    处理每行第一个数据
					int LastSum = SumofArray_C(ColValue, Radius * 2 + 1);                  //    处理每行第一个数据
					if (LastSum != LastSum2) {
						LOGD("ss");
					}
					LinePD[0] = IM_ClampToByte(LastSum * Inv);

					int BlockSize = 4, Block = (Width - 1) / BlockSize;
					__m128i OldSum = _mm_set1_epi32(LastSum);
					__m128 Inv128 = _mm_set1_ps(Inv);

					int X = 1;

					for (; X + BlockSize <= Width; X += BlockSize)
					{
						__m128i ColValueOut = _mm_loadu_si128((__m128i*)(ColValue + X - 1));                      
						__m128i ColValueIn = _mm_loadu_si128((__m128i*)(ColValue + X + Radius + Radius));         
						__m128i ColValueDiff = _mm_sub_epi32(ColValueIn, ColValueOut);                            // [P3          P2       P1        P0]     
						__m128i Value_Temp = _mm_add_epi32(ColValueDiff, _mm_slli_si128(ColValueDiff, 4));        // [P3+P2       P2+P1    P1+P0     P0]
						__m128i Value = _mm_add_epi32(Value_Temp, _mm_slli_si128(Value_Temp, 8));                 // [P3+P2+P1+P0 P2+P1+P0 P1+P0     P0]
						__m128i NewSum = _mm_add_epi32(OldSum, Value);                                            // [总和,       后三项和, 后两项和, 末项]
						OldSum = _mm_shuffle_epi32(NewSum, _MM_SHUFFLE(3, 3, 3, 3));                              // _MM_SHUFFLE 将NewSum的最高位元素复制到所有通道
						__m128 Mean = _mm_mul_ps(_mm_cvtepi32_ps(NewSum), Inv128);
						_mm_storesi128_4char(_mm_cvtps_epi32(Mean), LinePD + X);
						/**
						* _mm_slli_si128(ColValueDiff, 4): 从低地址向高地址移动4个字节，也就是32位（ColValueDiff中的一个元素）；
						* _mm_set1_epi32(LastSum): 将整型LastSum复制到128位寄存器的全部4个32位通道
						* _mm_set1_ps(Inv): 将浮点型Inv复制到128位寄存器的全部4个32位浮点通道
						* _mm_storesi128_4char: 将浮点结果转换为8位像素值并存储
						*/
					}

					for (; X < Width; X++)
					{
						int NewSum = LastSum - ColValue[X - 1] + ColValue[X + Radius + Radius];
						LinePD[X] = IM_ClampToByte(NewSum * Inv);
						LastSum = NewSum;
					}

					//? 1 上面的for循环和下面的for循环计算处的结果并不一样；
					//? 2 第一列与原生算子有差异！
				}
			}
			else if (Channel == 3)
			{

			}
			else if (Channel == 4)
			{

			}
			free(ColValue);
			free(ColOffset);
			return 1;
		}

		int IM_BoxBlur_SSE_Blocks(cv::Mat& src, cv::Mat& dst, int radius, int blocks, int thread_nums)
		{
			if (src.rows % blocks != 0) { return -1; }
			if (thread_nums > blocks) { return -1; }

			dst = cv::Mat::zeros(src.size(), src.type());
			int block_height = src.rows / blocks;  // 计算每块高度

			// 分块处理图像
#pragma omp parallel for num_threads(thread_nums)
			for (int j = 0; j < blocks; ++j) {
				// 计算当前块的起始行和高度
				int start_row = j * block_height;
				int actual_height = (j == blocks - 1) ? (src.rows - (blocks-1) * block_height) : block_height;

				// 获取当前块的输入/输出指针
				uchar* src_block = src.ptr<uchar>(start_row);
				uchar* dst_block = dst.ptr<uchar>(start_row);

				// 处理当前分块
				int result2 = IM_BoxBlur_SSE(
					src_block,
					dst_block,
					src.cols,    // 保持原宽度
					actual_height,  // 当前分块高度
					src.cols,     // 保持原步长
					radius
				);
			}

			return 1;
		}

		int IM_BoxBlur_SSE2(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int Radius)
		{
			int Channel = Stride / Width;
			if ((Src == NULL) || (Dest == NULL))
				return 0;
			if ((Width <= 0) || (Height <= 0) || (Radius <= 0))
				return 0;
			if ((Channel != 1) && (Channel != 3) && (Channel != 4))
				return 0;


			Radius = (std::min)((std::min)(Radius, Width - 1), Height - 1);        //    由于镜像的需求，要求半径不能大于宽度或高度-1的数据
			int SampleAmount = (2 * Radius + 1) * (2 * Radius + 1);
			float Inv = 1.0 / SampleAmount;

			int* ColValue = (int*)malloc((Width + Radius + Radius) * (Channel == 1 ? Channel : 4) * sizeof(int));
			int* ColOffset = (int*)malloc((Height + Radius + Radius) * sizeof(int));


			for (int Y = 0; Y < Height + Radius + Radius; Y++)
				ColOffset[Y] = IM_GetMirrorPos(Height, Y - Radius);

			if (Channel == 1)
			{
				for (int Y = 0; Y < Height; Y++)
				{
					unsigned char* LinePD = Dest + Y * Stride;
					if (Y == 0)
					{
						memset(ColValue + Radius, 0, Width * sizeof(int));
						for (int Z = -Radius; Z <= Radius; Z++)
						{
							unsigned char* LinePS = Src + ColOffset[ Y + Z + Radius] * Stride;

							int BlockSize = 8, Block = Width / BlockSize;
							int X = 0;
							for (; X < Width; X += BlockSize)
							{
								int* DestP = ColValue + X + Radius;
								__m128i Sample = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(LinePS + X)));
								_mm_storeu_si128((__m128i*)DestP, _mm_add_epi32(_mm_loadu_si128((__m128i*)DestP), _mm_cvtepi16_epi32(Sample)));
								_mm_storeu_si128((__m128i*)(DestP + 4), _mm_add_epi32(_mm_loadu_si128((__m128i*)(DestP + 4)), _mm_unpackhi_epi16(Sample, _mm_setzero_si128())));
							}

							for (; X < Width; X++)
							{
								ColValue[X + Radius] += LinePS[X];                                            //    更新列数据
							}
						}
					}
					else
					{
						unsigned char* RowMoveOut = Src + ColOffset[Y - 1] * Stride;                //    即将减去的那一行的首地址
						unsigned char* RowMoveIn = Src + ColOffset[Y + Radius + Radius] * Stride;    //    即将加上的那一行的首地址

						int BlockSize = 8, Block = Width / BlockSize;
						__m128i Zero = _mm_setzero_si128();
						int X = 0;
						for (; X < Width; X += BlockSize)
						{
							int* DestP = ColValue + X + Radius;
							__m128i MoveOut = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(RowMoveOut + X)), Zero);
							__m128i MoveIn = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(RowMoveIn + X)), Zero);
							__m128i Diff = _mm_sub_epi16(MoveIn, MoveOut);                        //    注意这个有负数也有正数的，有负数时转换为32位是不能用_mm_unpackxx_epi16体系的函数
							_mm_storeu_si128((__m128i*)DestP, _mm_add_epi32(_mm_loadu_si128((__m128i*)DestP), _mm_cvtepi16_epi32(Diff)));
							_mm_storeu_si128((__m128i*)(DestP + 4), _mm_add_epi32(_mm_loadu_si128((__m128i*)(DestP + 4)), _mm_cvtepi16_epi32(_mm_srli_si128(Diff, 8))));
						}
						for (; X < Width; X++)
						{
							ColValue[X + Radius] -= RowMoveOut[X] - RowMoveIn[X];                                            //    更新列数据
						}
					}

					//for (int i = 0; i < Width + Radius + Radius; i++) {
					//	LOGD("ColValue[{}] = {};", i, ColValue[i]);
					//}

					//FillLeftAndRight_Mirror_SSE(ColValue, Width, Radius);                  //    镜像填充左右数据
					//for (int i = 0; i < Width + Radius + Radius; i++) {
					//	LOGD("ColValue[{}] = {};", i, ColValue[i]);
					//}

					int LastSum = SumofArray_C(ColValue, Radius * 2 + 1);                  //    处理每行第一个数据
					LinePD[0] = IM_ClampToByte(LastSum * Inv);

					int BlockSize = 4, Block = (Width - 1) / BlockSize;
					__m128i OldSum = _mm_set1_epi32(LastSum);
					__m128 Inv128 = _mm_set1_ps(Inv);

					int X = 1;

					for (; X < Width; X += BlockSize)
					{
						__m128i ColValueOut = _mm_loadu_si128((__m128i*)(ColValue + X - 1));
						__m128i ColValueIn = _mm_loadu_si128((__m128i*)(ColValue + X + Radius + Radius));
						__m128i ColValueDiff = _mm_sub_epi32(ColValueIn, ColValueOut);                            //    P3 P2 P1 P0                                                
						__m128i Value_Temp = _mm_add_epi32(ColValueDiff, _mm_slli_si128(ColValueDiff, 4));        //    P3+P2 P2+P1 P1+P0 P0
						__m128i Value = _mm_add_epi32(Value_Temp, _mm_slli_si128(Value_Temp, 8));                 //    P3+P2+P1+P0 P2+P1+P0 P1+P0 P0
						__m128i NewSum = _mm_add_epi32(OldSum, Value);
						OldSum = _mm_shuffle_epi32(NewSum, _MM_SHUFFLE(3, 3, 3, 3));                              //    重新赋值为最新值
						__m128 Mean = _mm_mul_ps(_mm_cvtepi32_ps(NewSum), Inv128);
						_mm_storesi128_4char(_mm_cvtps_epi32(Mean), LinePD + X);
					}



					for (; X < Width; X++)
					{
						int NewSum = LastSum - ColValue[X - 1] + ColValue[X + Radius + Radius];
						LinePD[X] = IM_ClampToByte(NewSum * Inv);
						LastSum = NewSum;
					}

					//free(ColValue);
				}
			}
			else if (Channel == 3)
			{

			}
			else if (Channel == 4)
			{

			}
			free(ColOffset);
			return 1;
		}

	};

	void RGB2Y_4(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int threads_num = 4) {
		const int B_WT = int(0.114 * 256 + 0.5);
		const int G_WT = int(0.587 * 256 + 0.5);
		const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)

#pragma omp parallel for num_threads(threads_num)
		for (int Y = 0; Y < Height; Y++) {
			unsigned char* LinePS = Src + Y * Stride;
			unsigned char* LinePD = Dest + Y * Width;
			int X = 0;

			// 每次处理12个像素：加载数据并进行运算
			for (; X < Width - 12; X += 12, LinePS += 36) {
				// 每个加载操作使用_mm_loadu_si128来加载16字节的数据
				// _mm_cvtepu8_epi16将8位无符号整数扩展为16位，以便进行乘法运算
				// _mm_loadu_si128 用于从内存加载数据到SIMD寄存器
				// (__m128i*)(LinePS + 0) 将指针LinePS转换为__m128i*类型。__m128i是一个128位的整数类型，可以容纳16个8位整数、8个16位整数、4个32位整数等。
				// 这里的转换告诉编译器，将LinePS指向的内存当作128位的数据来处理。

				__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 0))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); //1
				__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 1))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); //2
				__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 2))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); //3

				__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 8))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//4
				__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 9))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//5
				__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 10))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//6

				__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 18))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//7
				__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 19))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//8
				__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 20))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//9

				__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 26))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//10
				__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 27))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//11
				__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(LinePS + 28))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//12

				// 加法操作，将三个乘积结果相加，得到总和
				__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));//13
				__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));//14
				__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));//15
				__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));//16

				// 右移8位，相当于除以256，得到最终的灰度值
				__m128i sclaL = _mm_srli_epi16(sumaL, 8);//17
				__m128i sclaH = _mm_srli_epi16(sumaH, 8);//18
				__m128i sclbL = _mm_srli_epi16(sumbL, 8);//19
				__m128i sclbH = _mm_srli_epi16(sumbH, 8);//20

				// 使用_mm_shuffle_epi8进行数据的重新排列，将结果打包到正确的位置，
				__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//21   16个元素
				__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//22
				__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));//23
				__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));//24

				__m128i accumL = _mm_or_si128(shftaL, shftbL);//25
				__m128i accumH = _mm_or_si128(shftaH, shftbH);//26

				__m128i h3 = _mm_or_si128(accumL, accumH);//27

				// 最后存储到目标内存中
				_mm_storeu_si128((__m128i*)(LinePD + X), h3);
			}
			for (; X < Width; X++, LinePS += 3) {
				LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
			}
		}
	}

	// ===============================================================================================

	// for + cv::max(p1, p2)
	void test_doing(cv::Mat imgori1, cv::Mat imgori2) {
		cv::cvtColor(imgori1, imgori1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(imgori2, imgori2, cv::COLOR_BGR2GRAY);


		cv::Mat _grayA1, _grayB1;
		cv::Mat _grayA2, _grayB2;
		cv::blur(imgori1, _grayA1, cv::Size(21, 21));  //! 均值提取低频信息
		cv::blur(imgori2, _grayB1, cv::Size(21, 21));  //! 均值提取低频信息
		_grayA2 = _grayA1 - imgori1;
		_grayB2 = _grayB1 - imgori2;

		cv::Mat result(imgori1.rows, imgori1.cols, CV_8UC1);
		int total_pics_num = 1000;
		auto start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < total_pics_num; i++) {
			auto* p1 = _grayA2.ptr<unsigned char>();
			auto* p2 = _grayB2.ptr<unsigned char>();
			auto* presult = result.ptr<unsigned char>();

			const int total = imgori1.total();
#pragma omp parallel for num_threads(8)
			for (int i = 0; i < total; ++i) {
				presult[i] = (cv::max)(p1[i], p2[i]);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}

	// unsigned char
	void test_doing3(cv::Mat imgori1, cv::Mat imgori2) {
		cv::cvtColor(imgori1, imgori1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(imgori2, imgori2, cv::COLOR_BGR2GRAY);


		cv::Mat _grayA1, _grayB1;
		cv::Mat _grayA2, _grayB2;
		cv::blur(imgori1, _grayA1, cv::Size(21, 21));  //! 均值提取低频信息
		cv::blur(imgori2, _grayB1, cv::Size(21, 21));  //! 均值提取低频信息
		_grayA2 = _grayA1 - imgori1;
		_grayB2 = _grayB1 - imgori2;

		cv::Mat result(imgori1.rows, imgori1.cols, CV_8UC1);
		int total_pics_num = 1000;
		auto start = std::chrono::high_resolution_clock::now();

		/** 将cv::Mat的数据装入到数组中 */
		unsigned char* data_array1 = new unsigned char[_grayA2.total()];
		unsigned char* data_array2 = new unsigned char[_grayB2.total()];
		if (_grayA2.isContinuous() && _grayB2.isContinuous()) {
			memcpy(data_array1, _grayA2.data, _grayA2.total() * sizeof(unsigned char));
			memcpy(data_array2, _grayB2.data, _grayB2.total() * sizeof(unsigned char));
		}
		unsigned char* data_array3 = new unsigned char[_grayA2.total()];

		for (int i = 0; i < total_pics_num; i++) {
			/** min */
#pragma omp parallel for num_threads(8)
			for (int i = 0; i < _grayA2.total(); ++i) {
				data_array3[i] = (std::min)(data_array1[i], data_array2[i]);
			}
		}

		/** 将数组转换成mat */
		cv::Mat mat_deep(imgori1.rows, imgori1.cols, CV_8UC1);
		memcpy(mat_deep.data, data_array3, imgori1.rows * imgori1.cols * sizeof(unsigned char));


		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}

	// cv::max(mat1, mat2)
	void test_doing4(cv::Mat imgori1, cv::Mat imgori2) {
		cv::cvtColor(imgori1, imgori1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(imgori2, imgori2, cv::COLOR_BGR2GRAY);


		cv::Mat _grayA1, _grayB1;
		cv::Mat _grayA2, _grayB2;
		cv::blur(imgori1, _grayA1, cv::Size(21, 21));  //! 均值提取低频信息
		cv::blur(imgori2, _grayB1, cv::Size(21, 21));  //! 均值提取低频信息
		_grayA2 = _grayA1 - imgori1;
		_grayB2 = _grayB1 - imgori2;

		// cv::min
		cv::Mat result(imgori1.rows, imgori1.cols, CV_8UC1);
		int total_pics_num = 1000;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			result = (cv::min)(imgori1, imgori2);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}

	/** 测试两张图合并为相同shape的一张图的合并操作的时间 */
	void experiment1(std::vector<cv::Mat> input)
	{
		cv::Mat src1 = input[0];
		cv::Mat src2 = input[1];

		int total_pics_num = 1;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			test_doing(src1, src2);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

		return;
	}

	// 行优先遍历测试
	void rowMajorAccess(cv::Mat& img) {
		for (int r = 0; r < img.rows; ++r) {
			auto* ptr = img.ptr<uchar>(r);
			for (int c = 0; c < img.cols; ++c) {
				ptr[c] = static_cast<uchar>((ptr[c] + 1) % 256);
				//img.at<uchar>(r, c) = static_cast<uchar>((img.at<uchar>(r, c) + 1) % 256);
			}
		}
	}

	// 列优先遍历测试
	void colMajorAccess(cv::Mat& img) {
		for (int c = 0; c < img.cols; ++c) {
			for (int r = 0; r < img.rows; ++r) {
				img.at<uchar>(r, c) = static_cast<uchar>((img.at<uchar>(r, c) + 1) % 256);
			}
		}
	}

	/** 测试: 一张图像，先行后列遍历和先列后行遍历的差别 */
	void experiment2(std::vector<cv::Mat> input)
	{
		cv::Mat src1 = input[0];
		cv::Mat src2 = input[1];

		int total_pics_num = 1000;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			colMajorAccess(src1);
			//rowMajorAccess(src1);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			//colMajorAccess(src1);
			rowMajorAccess(src1);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

		return;
	}

	void hconcat_memcpy(cv::Mat& img1, cv::Mat& result, int i, int thread_num) {
#pragma omp parallel for num_threads(thread_num)
		for (int r = 0; r < img1.rows; ++r) {
			uchar* dst = result.ptr<uchar>(r);
			const uchar* src1 = img1.ptr<uchar>(r);
			memcpy(dst + i * img1.cols * img1.elemSize(), src1, img1.cols * img1.elemSize());
		}
	}

	cv::Mat hconcat_roi(cv::Mat& img1, cv::Mat& img2) {
		CV_Assert(img1.rows == img2.rows && img1.type() == img2.type());

		cv::Mat result(img1.rows, img1.cols + img2.cols, img1.type());

		// 左半部分
		img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));
		// 右半部分
		img2.copyTo(result(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

		return result;
	}

	/** 测试: 两张图像横向拼接在一起 */
	void experiment3(std::vector<cv::Mat> input)
	{
		cv::Mat src1 = input[0];
		cv::Mat src2 = input[1];

		int total_pics_num = 1000;
		cv::Mat result(input[0].rows, input[0].cols * input.size(), input[0].type());

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				hconcat_memcpy(input[j], result, j, 0);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				hconcat_memcpy(input[j], result, j, 2);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				hconcat_memcpy(input[j], result, j, 4);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				hconcat_memcpy(input[j], result, j, 8);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				hconcat_memcpy(input[j], result, j, 16);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));




		//start = std::chrono::high_resolution_clock::now();
		//for (int i = 0; i < total_pics_num; i++) {
		//	for (int j = 0; j < input.size(); j++)
		//	{
		//		if (j == 0) { result = input[0]; }
		//		else {
		//			//result = hconcat_roi(result, input[j]);
		//		}
		//	}
		//}
		//end = std::chrono::high_resolution_clock::now();
		//duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		//LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		//start = std::chrono::high_resolution_clock::now();
		//for (int i = 0; i < total_pics_num; i++) {

		//	for (int j = 0; j < input.size(); j++)
		//	{
		//		if (j == 0) { result = input[0]; }
		//		else {
		//			//cv::hconcat(result, input[j], result);
		//		}
		//	}
		//}
		//end = std::chrono::high_resolution_clock::now();
		//duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		//LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

		return;
	}

	int picshadowx(cv::Mat binary, cv::Mat* show, int numThreads = 4)
	{
		int _res = 2;
		int* blackcout = new int[binary.cols];
		memset(blackcout, 0, binary.cols * 4);
		int _total = 0;

#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < binary.rows; i++)
		{
			for (int j = 0; j < binary.cols; j++)
			{
				if (binary.at<uchar>(i, j) > 50)
				{
					blackcout[j]++; //垂直投影按列在x轴进行投影
					++_total;
				}
			}
		}
		double _avg = _total * 1.0 / binary.cols;
#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < binary.cols; i++)
		{
			if (blackcout[i] > _avg + 5.3)
			{
				_res = 1;
				break;
			}
		}


		int threadID = omp_get_thread_num();
		int startRow = (threadID * binary.rows) / numThreads;
		int endRow = ((threadID + 1) * binary.rows) / numThreads;
		if (nullptr != show)
		{
#pragma omp parallel for num_threads(numThreads)
			for (int i = 0; i < binary.cols; i++)
			{
				if (blackcout[i] > _avg + 5.3)
				{
					for (int j = 0; j < binary.rows / 10; j++)
					{
						show->at<cv::Vec3b>(j, i)[0] = 128;//翻转到下面，便于观看
						show->at<cv::Vec3b>(j, i)[1] = 128;//翻转到下面，便于观看
						show->at<cv::Vec3b>(j, i)[2] = 128;//翻转到下面，便于观看
					}
				}
			}
#pragma omp parallel for num_threads(numThreads)
			for (int i = 0; i < binary.cols; i++)
			{
				int count = blackcout[i];
				for (int j = 0; j < count; j++)
				{
					show->at<cv::Vec3b>(show->rows - 1 - j, i)[0] = 255; //翻转到下面，便于观看
					show->at<cv::Vec3b>(show->rows - 1 - j, i)[1] = 255; //翻转到下面，便于观看
					show->at<cv::Vec3b>(show->rows - 1 - j, i)[2] = 0;   //翻转到下面，便于观看
				}
			}
		}

		delete[] blackcout;
		blackcout = nullptr;

		return _res;
	}

	int picshadowx_ptr_0519(unsigned char* binary, unsigned char* show, int numThreads, int width, int height)
	{
		int _res = 1;

		int* feature1out = new int[width];  // 单列目标像素点数量（皮屑）
		int* feature2out = new int[width];  // 单列目标像素点区间的数量（气泡）
		int* feature3out = new int[width];  // 单列光纤宽度（凹陷和凸起）

		memset(feature1out, 0, width * 4);
		memset(feature2out, 0, width * 4);
		memset(feature3out, 0, width * 4);

		int _feature1_total = 0;
		int _feature2_total = 0;
		int _feature3_total = 0;

		try {

			/*** Step1: 计算每列的特征123 ***/
#pragma omp parallel for num_threads(numThreads) reduction(+:_feature1_total, _feature2_total, _feature3_total)
			// reduction子句会为每个线程创建一个私有的_total副本，在并行区域结束时将这些副本安全地累加到原始的_total变量中
			for (int j = 0; j < width; j++)
			{
				std::vector<std::pair<int, int>> feature2_result;
				int feature2_start = -1;
				int feature2_count = 0;

				int feature3_start = -1;
				int feature3_end = -1;

				for (int i = 0; i < height; i++)
				{
					int p = binary[i * width + j];

					// feature1：计算单列目标像素点数量
					if (p > FEATURE1_CONTRAST_DEGREE)
					{
#pragma omp atomic  // 并行安全
						feature1out[j]++;     //垂直投影按列在x轴进行投影
						++_feature1_total;
					}



					// feature2：计算目标像素点区间
					if (p > FEATURE2_CONTRAST_DEGREE) {
						if (feature2_start == -1) { feature2_start = i; } // 记录区间开始位置
						feature2_count++;
					}
					else {
						if (feature2_count >= FEATURE2_BLOCK_LENGTH) { feature2_result.emplace_back(feature2_start, i - 1); }
						feature2_start = -1;
						feature2_count = 0;
					}
					if (i == height - 1) {  // 处理末尾可能存在的连续目标像素点
						if (feature2_count >= FEATURE2_BLOCK_LENGTH) { feature2_result.emplace_back(feature2_start, i); }
					}



					// feature3: 光纤当前列的起始索引
					if (p > FEATURE3_CONTRAST_DEGREE && feature3_start == -1) { feature3_start = i; }
					if (p > FEATURE3_CONTRAST_DEGREE) { feature3_end = i; }

				}


				if (feature2_result.size() >= FEATURE2_BLOCK_NUMS) {
					int size = feature2_result.size();
					feature2out[j] = size;
					_feature2_total++;
				}

				int diff = std::abs(feature3_end - feature3_start - FEATURE3_DIAMETER);
				if (diff > FEATURE3_DIAMETER_THLD) {
					int value = diff - FEATURE3_DIAMETER_THLD;
					feature3out[j] = value;
					_feature3_total++;
				}
			}




			/* Step2: 根据规则（阴影高度、目标像素连续区间长度、列宽度）构建各特征对应缺陷区间 */
			double _avg = _feature1_total * 1.0 / width;

			std::vector<std::vector<int>> feature1_candidates_result;
			std::vector<int> feature1_temp_area;

			for (int i = 0; i < width; ++i) {
				bool condition1 = feature1out[i] - _avg > DEFECT_LENGTH;
				bool condition2 = feature1out[i] - _avg > DEFECT_LENGTH / 5;
				if ((condition1) || ((condition2) && !feature1_temp_area.empty()))
				{
					feature1_temp_area.push_back(i);
				}
				else
				{
					if (!feature1_temp_area.empty()) {  // 如果temp不为空，
						feature1_candidates_result.push_back(feature1_temp_area);
						feature1_temp_area.clear();
					}
				}
				if (i == width - 1 && !feature1_temp_area.empty()) { feature1_candidates_result.push_back(feature1_temp_area); feature1_temp_area.clear(); }  // 处理最后一个连续区间
			}


			std::vector<std::vector<int>> feature2_candidates_result;
			std::vector<int> feature2_temp_area;

			for (int i = 0; i < width; ++i) {
				bool condition1 = feature2out[i] > 0;
				if (condition1)
				{
					feature2_temp_area.push_back(i);
				}
				else
				{
					if (!feature2_temp_area.empty()) {  // 如果temp不为空，
						feature2_candidates_result.push_back(feature2_temp_area);
						feature2_temp_area.clear();
					}
				}

				if (i == width - 1 && !feature2_temp_area.empty()) { feature2_candidates_result.push_back(feature2_temp_area); feature2_temp_area.clear(); }  // 处理最后一个连续区间
			}


			std::vector<std::vector<int>> feature3_candidates_result;
			std::vector<int> feature3_temp_area;

			for (int i = 0; i < width; ++i) {
				bool condition1 = feature3out[i] > 0;
				if (condition1)
				{
					feature3_temp_area.push_back(i);
				}
				else
				{
					if (!feature3_temp_area.empty()) {  // 如果temp不为空，
						feature3_candidates_result.push_back(feature3_temp_area);
						feature3_temp_area.clear();
					}
				}

				if (i == width - 1 && !feature3_temp_area.empty()) { feature3_candidates_result.push_back(feature3_temp_area); feature3_temp_area.clear(); }  // 处理最后一个连续区间
			}




			/* Step3: 处理每个集合，根据规则(连续长度)判断该集合是否为缺陷区域 */
			int candidates_total_length = 0;

			std::vector<std::vector<int>> defects1_result;
			std::vector<std::vector<int>> defects2_result;
			std::vector<std::vector<int>> defects3_result;

			for (int i = 0; i < feature1_candidates_result.size(); ++i)
			{
				candidates_total_length += feature1_candidates_result[i].size();
				if (feature1_candidates_result[i].size() > DEFECT1_WIDTH) { defects1_result.push_back(feature1_candidates_result[i]); }
			}
			if (candidates_total_length > DEFECT1_WIDTH) {
				defects1_result = feature1_candidates_result;
			}

			for (int i = 0; i < feature2_candidates_result.size(); ++i)
			{
				if (feature2_candidates_result[i].size() > DEFECT2_WIDTH) { defects2_result.push_back(feature2_candidates_result[i]); }
			}

			for (int i = 0; i < feature3_candidates_result.size(); ++i)
			{
				if (feature3_candidates_result[i].size() > DEFECT3_WIDTH) { defects3_result.push_back(feature3_candidates_result[i]); }
			}




			/*** Step4: 判断是否NG ***/
			if (defects1_result.size() != 0) { _res = 2; }
			if (defects2_result.size() != 0) { _res = 2; }
			if (defects3_result.size() != 0) { _res = 2; }
			//LOGD("[MARK]: defects1_result.size: {};", defects1_result.size());
			//LOGD("[MARK]: defects2_result.size: {};", defects2_result.size());
			//LOGD("[MARK]: defects3_result.size: {};", defects3_result.size());





			/*** Step5: SHOW ***/
			if (nullptr != show)
			{

				/* 绘制超过缺陷长度阈值的列，及其超过数值 */
#pragma omp parallel for num_threads(numThreads)
				for (int i = 0; i < width; i++)
				{
					double temp = feature1out[i] - _avg - DEFECT_LENGTH;
					if (temp)
					{
						int temp2 = (std::min)(static_cast<int>(temp), DRAW_MAX_H);  //!? 限制绘制的高度
						temp2 = std::clamp(temp2, 0, height - 1 - NG_DRAW_MAX_H);
						for (int j = NG_DRAW_MAX_H; j < NG_DRAW_MAX_H + temp2; j++)
						{
							show[(j * width + i) * 3 + 0] = 128;
							show[(j * width + i) * 3 + 1] = 128;
							show[(j * width + i) * 3 + 2] = 128;
						}
					}
				}

				/* 绘制每列暗黑数目 */
#pragma omp parallel for num_threads(numThreads)
				for (int i = 0; i < width; i++)
				{
					for (int j = 0; j < feature1out[i]; j++)
					{
						int row = height - j - 1;

						show[(row * width + i) * 3 + 0] = 255;
						show[(row * width + i) * 3 + 1] = 255;
						show[(row * width + i) * 3 + 2] = 0;
					}
				}
			}

			/* 绘制被认定为缺陷的区域（连续列的集合） */
			for (int i = 0; i < defects1_result.size(); ++i)
			{
				for (int j = 0; j < defects1_result[i].size(); ++j)
				{
					int z = defects1_result[i][j];
					for (int j = 0; j < NG_DRAW_MAX_H; j++)
					{
						show[(j * width + z) * 3 + 0] = 0;
						show[(j * width + z) * 3 + 1] = 0;
						show[(j * width + z) * 3 + 2] = 255;
					}
				}
			}


			for (int i = 0; i < defects2_result.size(); ++i)
			{
				for (int j = 0; j < defects2_result[i].size(); ++j)
				{
					int c = defects2_result[i][j];
					int temp_draw_max = std::clamp(feature2out[c], 0, NG_DRAW_MAX_H);
					for (int z = NG_DRAW_MAX_H; z < NG_DRAW_MAX_H + temp_draw_max; z++)
					{
						show[(z * width + c) * 3 + 0] = 255;
						show[(z * width + c) * 3 + 1] = 0;
						show[(z * width + c) * 3 + 2] = 0;
					}
				}
			}


			for (int i = 0; i < defects3_result.size(); ++i)
			{
				for (int j = 0; j < defects3_result[i].size(); ++j)
				{
					int c = defects3_result[i][j];
					int temp_draw_max = std::clamp(feature3out[c], 0, NG_DRAW_MAX_H);
					for (int z = NG_DRAW_MAX_H * 2; z < NG_DRAW_MAX_H * 2 + temp_draw_max; z++)
					{
						show[(z * width + c) * 3 + 0] = 0;
						show[(z * width + c) * 3 + 1] = 255;
						show[(z * width + c) * 3 + 2] = 0;
					}
				}
			}



			/* 绘制avg向下取整值的位置 */
			int int_avg = height - static_cast<int>(std::floor(_avg));  // i = 3 当_avg为0时，
			int_avg = std::clamp(int_avg, 0, height - 1);               // 没有该句，会越界 int_avg = height。
			for (int col = 0; col < width; ++col) {
				//int int_avg =10;  // i = 3
				show[(int_avg * width + col) * 3 + 0] = 0;
				show[(int_avg * width + col) * 3 + 1] = 0;
				show[(int_avg * width + col) * 3 + 2] = 255;
			}
		}
		catch (const cv::Exception& e)
		{
			LOGE("NG_UNDEFINED: e1: {};", e.what());
		}
		catch (const std::exception& e)
		{
			LOGE("NG_UNDEFINED: e2: {};", e.what());
		}
		catch (...)
		{
			LOGE("NG_UNDEFINED: e3: unkown;");
		}


		delete[] feature1out;
		delete[] feature2out;
		delete[] feature3out;

		feature1out = nullptr;
		feature2out = nullptr;
		feature3out = nullptr;

		return _res;
	}

	void experiment4(std::vector<cv::Mat> input) {
		cv::Mat src1 = input[0];

		int total_pics_num = 1000;
		cv::Mat result(input[0].rows, input[0].cols * input.size(), input[0].type());

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				hconcat_memcpy(input[j], result, j, 4);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));



		cv::Mat imgrst = result.clone();
		BlurVersion2 blur2 = BlurVersion2();
		cv::Mat _gray;
		cv::Mat  _gray2;

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			/** 1 BGR2GRAY*/
			cv::cvtColor(result, _gray, cv::COLOR_BGR2GRAY);  // 希望取消该方式!

			//int Height = result.rows;
			//int Width = result.cols;
			//int Stride = Width * 3;
			//unsigned char* Src = result.data;
			//unsigned char* Dest = new unsigned char[Height * Width];  //! 输出缓冲区需要预先分配 Width*Height 字节空间
			//RGB2Y_4(Src, Dest, Width, Height, Stride, 2);     // sse 一次处理12个
			//_gray = cv::Mat(Height, Width, CV_8UC1, Dest);  // 基本不消耗时间


			/** 2 BoxFilter */
			//cv::blur(_gray, _gray2, cv::Size(21, 21));

			//_gray2 = cv::Mat::zeros(_gray.size(), _gray.type());
			//int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _gray2.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);

			blur2.IM_BoxBlur_SSE_Blocks(_gray, _gray2, 10, 2, 2);
			//! 经验： 当算法中存在多个使用omp的算子时，需要合理分配omp的线程数目，不能太大，否则计算慢。


			/** 3 subtraction */
			_gray = _gray2 - _gray;  // 如果复用 gray 用时0.08ms, 而使用新矩阵会0.45ms;

			//cv::Mat subtraction1 = standard_gray3 - _gray3;
			//cv::Mat subtraction2 = _gray3 - standard_gray3;


			///** 4 picshadowy */
			int _iresult = picshadowx(_gray, &imgrst, 2);



			//LOGD("s");

		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		cv::Mat imgrstB = src1.clone();
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < input.size(); j++)
			{
				/** 初始化 */
				cv::Mat _grayB, _grayB2;


				/** BGR2GRAY */
				cv::cvtColor(input[j], _grayB, cv::COLOR_BGR2GRAY);

				//int Height = input[j].rows;
				//int Width = input[j].cols;
				//int Stride = Width * 3;
				//unsigned char* Src = input[j].data;
				//unsigned char* Dest = new unsigned char[Height * Width];  //! 输出缓冲区需要预先分配 Width*Height 字节空间
				//RGB2Y_4(Src, Dest, Width, Height, Stride, 2);       // sse 一次处理12个
				//_grayB = cv::Mat(Height, Width, CV_8UC1, Dest);  // 基本不消耗时间


				/** BoxFilter */
				//cv::Mat tem;
				//cv::blur(_grayB, tem, cv::Size(21, 21));
				//_grayB2 = cv::Mat::zeros(_grayB.size(), _grayB.type());
				//int result2 = blur2.IM_BoxBlur_SSE_Comment(_grayB.ptr<uchar>(0), _grayB2.ptr<uchar>(0), _grayB.cols, _grayB.rows, _grayB.cols, 10);
				blur2.IM_BoxBlur_SSE_Blocks(_grayB, _grayB2, 10, 2, 2);

				//cv::Mat sub1 = tem - _grayB2;
				//cv::Mat sub2 = _grayB2 - tem;

				/** subtraction */
				_grayB = _grayB2 - _grayB;


				/** picshadowy */
				int _iresult = picshadowx(_grayB, &imgrstB, 2);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}

	int picshadowy(cv::Mat binary, cv::Mat* show, int numThreads = 4, int single_rows = -1)
	{
		if (single_rows == -1) { single_rows = binary.rows; }
		int _res = 2;
		int* blackcout = new int[binary.rows];
		memset(blackcout, 0, binary.rows * 4);
		int _total = 0;

#pragma omp parallel for num_threads(numThreads)  reduction(+:_total)
		for (int i = 0; i < binary.rows; i++)
		{
			if (single_rows > 0 && i % single_rows < 5) { continue; }  // 避免拼接交界处出现的伪缺陷问题

			auto* ptr = binary.ptr<uchar>(i);
			for (int j = 0; j < binary.cols; j++)
			{
				if (ptr[j] > 50)
				{
					#pragma omp atomic  // 并行安全
					blackcout[i]++; //垂直投影按列在x轴进行投影
					++_total;
				}
			}
		}
		double _avg = _total * 1.0 / binary.rows;



#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < binary.rows; i++)
		{
			if (single_rows > 0 && i % single_rows < 5) { continue; }  // 避免拼接交界处出现的伪缺陷问题

			if (blackcout[i] > _avg + 5.3)
			{
				_res = 1;
				break;
			}
		}


		if (nullptr != show)
		{
#pragma omp parallel for num_threads(numThreads)
			for (int i = 0; i < binary.rows; i++)
			{
				if (blackcout[i] > _avg + 5.3)
				{
					for (int j = 0; j < binary.cols / 10; j++)
					{
						show->at<cv::Vec3b>(i, j)[0] = 128;//翻转到下面，便于观看
						show->at<cv::Vec3b>(i, j)[1] = 128;//翻转到下面，便于观看
						show->at<cv::Vec3b>(i, j)[2] = 128;//翻转到下面，便于观看
					}
				}
			}
#pragma omp parallel for num_threads(numThreads)
			for (int i = 0; i < binary.rows; i++)
			{
				int count = blackcout[i];
				for (int j = 0; j < count; j++)
				{
					show->at<cv::Vec3b>(i, show->cols - 1 - j)[0] = 255; //翻转到下面，便于观看
					show->at<cv::Vec3b>(i, show->cols - 1 - j)[1] = 255; //翻转到下面，便于观看
					show->at<cv::Vec3b>(i, show->cols - 1 - j)[2] = 0;   //翻转到下面，便于观看
				}
			}
		}

		delete[] blackcout;
		blackcout = nullptr;

		return _res;
	}

	void ff(std::vector<cv::Mat>& input, std::vector<cv::Mat>& output)
	{
		for (int i = 0; i < input.size(); i++)
		{
			cv::Mat dst;
			cv::rotate(input[i], dst, cv::ROTATE_90_CLOCKWISE);
			output.push_back(dst.isContinuous() ? dst : dst.clone());
		}

	}

	void vconcat_memcpy(cv::Mat& img1, cv::Mat& result, int i, int thread_num) {
		const size_t row_bytes = img1.cols * img1.elemSize();

#pragma omp parallel for num_threads(thread_num)
		for (int r = 0; r < img1.rows; ++r) {
			uchar* dst = result.ptr<uchar>(img1.rows * i + r);
			const uchar* src = img1.ptr<uchar>(r);
			memcpy(dst, src, row_bytes);
		}

		//const size_t row_bytes = img1.cols * img1.elemSize() * img1.rows;
		//memcpy(result.data + i * img1.cols * img1.elemSize() * img1.rows, img1.data, row_bytes);

	}

	/** 输入需要是反转后的；且配备Y方向的投影函数 */
	void experiment5(std::vector<cv::Mat> input) {

		std::vector<cv::Mat> output;
		ff(input, output);

		cv::Mat src1 = output[0];

		int total_pics_num = 1000;
		cv::Mat result(output[0].rows * output.size(), output[0].cols , output[0].type());

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < output.size(); j++)
			{
				vconcat_memcpy(output[j], result, j, 4);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));






		/** standard process*/
		cv::Mat standard_gray, standard_gray2, standard_gray3;
		cv::Mat standard_imgrst = result.clone();
		cv::cvtColor(result, standard_gray, cv::COLOR_BGR2GRAY);  // 希望取消该方式!
		cv::blur(standard_gray, standard_gray2, cv::Size(21, 21));
		standard_gray3 = standard_gray2 - standard_gray;
		int standard_iresult = picshadowy(standard_gray3, &standard_imgrst, 4);



		/** experiments */
		BlurVersion2 blur2 = BlurVersion2();
		cv::Mat imgrst = result.clone();
		cv::Mat _gray;
		cv::Mat  _gray2;
		//cv::Mat _gray = standard_gray;
		//cv::blur(_gray, _gray2, cv::Size(21, 21));

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			/** 1 BGR2GRAY*/
			//cv::cvtColor(result, _gray, cv::COLOR_BGR2GRAY);  // 希望取消该方式!

			int Height = result.rows;
			int Width = result.cols;
			int Stride = Width * 3;
			unsigned char* Src = result.data;
			unsigned char* Dest = new unsigned char[Height * Width];  //! 输出缓冲区需要预先分配 Width*Height 字节空间
			RGB2Y_4(Src, Dest, Width, Height, Stride, 2);     // sse 一次处理12个
			_gray = cv::Mat(Height, Width, CV_8UC1, Dest);  // 基本不消耗时间


			/** 2 BoxFilter */
			//cv::blur(_gray, _gray2, cv::Size(21, 21));

			//_gray2 = cv::Mat::zeros(_gray.size(), _gray.type());
			//int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _gray2.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);

			blur2.IM_BoxBlur_SSE_Blocks(_gray, _gray2, 10, 10, 2); 
			//! 经验： 当算法中存在多个使用omp的算子时，需要合理分配omp的线程数目，不能太大，否则计算慢。


			/** 3 subtraction */
			_gray = _gray2 - _gray;  // 如果复用 gray 用时0.08ms, 而使用新矩阵会0.45ms;

			//cv::Mat subtraction1 = standard_gray3 - _gray3;
			//cv::Mat subtraction2 = _gray3 - standard_gray3;


			///** 4 picshadowy */
			int _iresult = picshadowy(_gray, &imgrst, 2, src1.rows);



			//LOGD("s");

		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		cv::Mat imgrstB = src1.clone();
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			for (int j = 0; j < output.size(); j++)
			{
				/** 初始化 */
				cv::Mat _grayB, _grayB2;


				/** BGR2GRAY */
				cv::cvtColor(output[j], _grayB, cv::COLOR_BGR2GRAY);

				//int Height = output[j].rows;
				//int Width = output[j].cols;
				//int Stride = Width * 3;
				//unsigned char* Src = output[j].data;
				//unsigned char* Dest = new unsigned char[Height * Width];  //! 输出缓冲区需要预先分配 Width*Height 字节空间
				//RGB2Y_4(Src, Dest, Width, Height, Stride, 2);       // sse 一次处理12个
				//_grayB = cv::Mat(Height, Width, CV_8UC1, Dest);  // 基本不消耗时间


				/** BoxFilter */
				//cv::blur(_grayB, _grayB2, cv::Size(21, 21));
				//_grayB2 = cv::Mat::zeros(_grayB.size(), _grayB.type());
				//int result2 = blur2.IM_BoxBlur_SSE(_grayB.ptr<uchar>(0), _grayB2.ptr<uchar>(0), _grayB.cols, _grayB.rows, _grayB.cols, 10);
				blur2.IM_BoxBlur_SSE_Blocks(_grayB, _grayB2, 10, 2, 2);



				/** subtraction */
				_grayB = _grayB2 - _grayB;


				/** picshadowy */
				int _iresult = picshadowy(_grayB, &imgrstB, 2);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}

	/** 学习boxfilter:  */
	void studyBoxFilter() {
		cv::Mat src(4, 5, CV_8UC1); // 创建 10x10 单通道矩阵
		//cv::randu(src, cv::Scalar(0), cv::Scalar(11)); // [0,11) 区间'
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				src.at<char>(i, j) = i * src.cols + j + 1; // 计算连续值
			}
		}
		/** 原始数据
		1  2  3  4  5
		6  7  8  9  10
		11 12 13 14 15
		16 17 18 19 20
		*/



		/** 填充后的数据
		7	6  7  8  9  10  9
		-----------------------
		2 | 1  2  3  4  5  | 4
		7 |	6  7  8  9  10 | 9
		12|	11 12 13 14 15 | 14
		17| 16 17 18 19 20 | 19
		-----------------------
		12	11 12 13 14 15   14
		*/


		cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
		BlurVersion2 blur2 = BlurVersion2();
		int result2 = blur2.IM_BoxBlur_SSE_Comment(src.ptr<uchar>(0), dst.ptr<uchar>(0), src.cols, src.rows, src.cols, 1);
		/**
		5  5  6  7  7
		6  7  8  9  9
		11   12  13  14  14
		13   13   14   15   16
		*/
		cv::Mat dst2;
		cv::blur(src, dst2, cv::Size(3, 3));
		
		LOGD("--");

	}


	void studySSECase1() {

		float op1[4] = { 1.0, 2.0, 3.0, 4.0 };
		float op2[4] = { 1.0, 2.0, 3.0, 4.0 };
		float result[4];

		__m128  a;
		__m128  b;
		__m128  c;

		// Load
		a = _mm_loadu_ps(op1);
		b = _mm_loadu_ps(op2);

		// Calculate
		c = _mm_add_ps(a, b);	// c = a + b

		// Store
		_mm_storeu_ps(result, c);

		cout << result[0] << endl;
		cout << result[1] << endl;
		cout << result[2] << endl;
		cout << result[3] << endl;
		system("pause");

	}


	void studySSECase2() {


		__declspec(align(16)) float op1[4] = { 1.0, 2.0, 3.0, 4.0 };
		__declspec(align(16)) float op2[4] = { 1.0, 2.0, 3.0, 4.0 };
		_MM_ALIGN16 float result[4];		// _MM_ALIGN16等同于__declspec(align(16))

		__m128  a;
		__m128  b;
		__m128  c;

		// Load
		a = _mm_load_ps(op1);
		b = _mm_load_ps(op2);

		// Calculate
		c = _mm_add_ps(a, b);	// c = a + b

		// Store
		_mm_store_ps(result, c);

		cout << result[0] << endl;
		cout << result[1] << endl;
		cout << result[2] << endl;
		cout << result[3] << endl;
		system("pause");

	}


	void sse_add(float* srcA, float* srcB, float* dest, int n) {
		int len = n >> 2;
		for (int i = 0; i < len; i++) {
			*(__m128*)(dest + i * 4) = _mm_add_ps(*(__m128*)(srcA + i * 4), *(__m128*)(srcB + i * 4));
		}
	}

	void normal_add(float* srcA, float* srcB, float* dest, int n) {
		for (int i = 0; i < n; i++) {
			dest[i] = srcA[i] + srcB[i];
		}
	}

	void studySSECase3() {



		DWORD timeStart = 0, timeEnd = 0;
		const int size = 10000; //申请的内存中存放的数据个数
		const int count = 10000;//循环计算的次数，便于观察执行效率

		// 分配16字节对齐的内存
		_MM_ALIGN16 float* srcA = (_MM_ALIGN16 float*)_mm_malloc(sizeof(float) * size, 16);
		_MM_ALIGN16 float* srcB = (_MM_ALIGN16 float*)_mm_malloc(sizeof(float) * size, 16);
		_MM_ALIGN16 float* dest = (_MM_ALIGN16 float*)_mm_malloc(sizeof(float) * size, 16);

		// 初始化
		for (int i = 0; i < size; i++) {
			srcA[i] = (float)i;
		}
		memcpy_s(srcB, sizeof(float) * size, srcA, sizeof(float) * size);

		// 标准加法
		timeStart = GetTickCount();
		for (int i = 0; i < count; i++) {
			normal_add(srcA, srcB, dest, size);
		}
		timeEnd = GetTickCount();
		cout << "标准加法" << (timeEnd - timeStart) << "毫秒" << endl;

		// SSE指令加法
		timeStart = GetTickCount();
		for (int i = 0; i < count; i++) {
			sse_add(srcA, srcB, dest, size);
		}
		timeEnd = GetTickCount();
		cout << "SSE加法" << (timeEnd - timeStart) << "毫秒" << endl;

		// 释放内存
		_mm_free(srcA);
		_mm_free(srcB);
		_mm_free(dest);

		system("pause");

	}

	// doing: 原始dong
	void baseline(cv::Mat& img1, cv::Mat& imgrst, BlurVersion2& blur2)
	{
		//try  // it really needs to exist.
		//{


		//}
		//catch (const cv::Exception& e)
		//{
		//	LOGE("NG_UNDEFINED: e1: {};", e.what());
		//}
		//catch (const std::exception& e)
		//{
		//	LOGE("NG_UNDEFINED: e2: {};", e.what());
		//}
		//catch (...)
		//{
		//	LOGE("NG_UNDEFINED: e3: unkown;");
		//}

					/** init */
		int _i = 0;
		cv::Mat _gray, _gray2;
		if (img1.channels() == 1)
		{
			cv::cvtColor(img1, imgrst, cv::COLOR_GRAY2BGR);
			img1.copyTo(_gray);
		}
		else
		{
			imgrst = img1.clone();
			cv::cvtColor(img1, _gray, cv::COLOR_BGR2GRAY);

			//int Height = data.imgori.rows;
			//int Width = data.imgori.cols;
			//unsigned char* Src = data.imgori.data;
			//unsigned char* Dest = new unsigned char[Height * Width];  //! 输出缓冲区需要预先分配 Width*Height 字节空间
			//int Stride = Width * 3;
			//RGB2Y_4(Src, Dest, Width, Height, Stride);      // sse 一次处理12个
			//_gray = cv::Mat(Height, Width, CV_8UC1, Dest);  // 基本不消耗时间
		}

		/** block1 */
		//_gray2 = _gray.clone();
		//cv::blur(_gray, _gray2, cv::Size(21, 21));
		_gray2 = cv::Mat::zeros(_gray.size(), _gray.type());
		int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _gray2.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);
		//blur2.IM_BoxBlur_SSE_Blocks(_gray, _gray2, 10, 2, 2);


		/** block2 */
		_gray = _gray2 - _gray; // 提取高频信息

		/** block3 */
		int _iresult = picshadowx(_gray, &imgrst, 4);

		return ;
	}


	void optimize_0519(cv::Mat& img1, cv::Mat& imgrst, BlurVersion2& blur2)
	{
		//try  // it really needs to exist.
		//{


		//}
		//catch (const cv::Exception& e)
		//{
		//	LOGE("NG_UNDEFINED: e1: {};", e.what());
		//}
		//catch (const std::exception& e)
		//{
		//	LOGE("NG_UNDEFINED: e2: {};", e.what());
		//}
		//catch (...)
		//{
		//	LOGE("NG_UNDEFINED: e3: unkown;");
		//}

					/** init */
		int _i = 0;
		cv::Mat _gray, _gray2;
		if (img1.channels() == 1)
		{
			cv::cvtColor(img1, imgrst, cv::COLOR_GRAY2BGR);
			img1.copyTo(_gray);
		}
		else
		{
			imgrst = img1.clone();
			cv::cvtColor(img1, _gray, cv::COLOR_BGR2GRAY);

			//int Height = data.imgori.rows;
			//int Width = data.imgori.cols;
			//unsigned char* Src = data.imgori.data;
			//unsigned char* Dest = new unsigned char[Height * Width];  //! 输出缓冲区需要预先分配 Width*Height 字节空间
			//int Stride = Width * 3;
			//RGB2Y_4(Src, Dest, Width, Height, Stride);      // sse 一次处理12个
			//_gray = cv::Mat(Height, Width, CV_8UC1, Dest);  // 基本不消耗时间
		}

		/** block1 */
		//_gray2 = _gray.clone();
		//cv::blur(_gray, _gray2, cv::Size(21, 21));
		_gray2 = cv::Mat::zeros(_gray.size(), _gray.type());
		int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _gray2.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);
		//blur2.IM_BoxBlur_SSE_Blocks(_gray, _gray2, 10, 2, 2);


		/** block2 */
		_gray = _gray2 - _gray; // 提取高频信息

		/** block3 */
		int _iresult = picshadowx_ptr_0519(_gray.ptr(0), imgrst.ptr(0), 4, _gray.cols, _gray.rows);

		return;
	}

	// 最原始的
	void experiment6(std::vector<cv::Mat> input) {
		auto start = std::chrono::high_resolution_clock::now();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		int total_pics_num = 1000;
		BlurVersion2 blur2 = BlurVersion2();


		/** standard process*/
		cv::Mat src1 = input[0];
		cv::Mat standard_gray, standard_gray2, standard_gray3;
		cv::Mat standard_imgrst = src1.clone();
		cv::cvtColor(src1, standard_gray, cv::COLOR_BGR2GRAY);  // 希望取消该方式!
		cv::blur(standard_gray, standard_gray2, cv::Size(21, 21));
		standard_gray3 = standard_gray2 - standard_gray;
		int standard_iresult = picshadowy(standard_gray3, &standard_imgrst, 4);



		/** experiments */
		//cv::Rect roi = cv::Rect(cv::Point(0, 1000), cv::Point(4095, 1200));
		//cv::Mat img1 = src1(roi);
		cv::Mat img1 = src1.clone();
		cv::Mat imgrst = img1.clone();
		baseline(img1, imgrst, blur2);

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			baseline(img1, imgrst, blur2);
			//if (i % 10 == 0) { LOGD(i); }
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}



	// picshadowx_ptr_0519测试
	void experiment7(std::vector<cv::Mat> input) {
		auto start = std::chrono::high_resolution_clock::now();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		int total_pics_num = 1000;
		BlurVersion2 blur2 = BlurVersion2();


		/** standard process*/
		cv::Mat src1 = input[0];
		cv::Mat standard_gray, standard_gray2, standard_gray3;
		cv::Mat standard_imgrst = src1.clone();
		cv::cvtColor(src1, standard_gray, cv::COLOR_BGR2GRAY);  // 希望取消该方式!
		cv::blur(standard_gray, standard_gray2, cv::Size(21, 21));
		standard_gray3 = standard_gray2 - standard_gray;
		int standard_iresult = picshadowy(standard_gray3, &standard_imgrst, 4);



		/** experiments */
		//cv::Rect roi = cv::Rect(cv::Point(0, 1000), cv::Point(4095, 1200));
		//cv::Mat img1 = src1(roi);
		cv::Mat img1 = src1.clone();
		cv::Mat imgrst = img1.clone();
		optimize_0519(img1, imgrst, blur2);

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			optimize_0519(img1, imgrst, blur2);
			//if (i % 10 == 0) { LOGD(i); }
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

	}


	int enhance_center_image(unsigned char* binary, int width, int height)
	{
		int FEATURE3_CONTRAST_DEGREE = 50;
#pragma omp parallel for num_threads(4)
		for (int j = 0; j < width; j++)
		{
			int feature3_start = -1;
			int feature3_end = -1;

			int roi_up = -1;
			int roi_down = -1;

			for (int i = 0; i < height - 3; i++)  //!? 与next_next_p的处理相联系
			{
				int p = binary[i * width + j];
				int next_p = binary[(i + 1) * width + j];
				int next_next_p = binary[(i + 2) * width + j];

				// 找到上边缘
				bool feature3_cond1 = (p > FEATURE3_CONTRAST_DEGREE && next_p > FEATURE3_CONTRAST_DEGREE && next_next_p > FEATURE3_CONTRAST_DEGREE);  //!? 必须满足连续一段的像素均超过某个均值，避免椒盐噪声的影响！
				if (feature3_cond1 && feature3_start == -1) { feature3_start = i; }  // 记录上边界
				if (feature3_cond1) { feature3_end = i; }                             // 记录下边界，一直在被更新

				// 确定中心区域
				if (feature3_start != -1) {
					roi_up = feature3_start + FEATURE3_DIAMETER / 4;
					roi_down = feature3_start + FEATURE3_DIAMETER / 4 * 3;
				}

				if (i >= roi_up && i <= roi_down) { binary[i * width + j] += 30; binary[i * width + j] -= 30;
				}
			}
		}


		return 0;
	}


	inline int enhance_center_image_0612(unsigned char* binary, int width, int height, int numThreads)
	{
		int FEATURE3_CONTRAST_DEGREE = 50;

		const int roi_up_offset = FEATURE3_DIAMETER / 4;
		const int roi_down_offset = FEATURE3_DIAMETER / 4 * 3;

#pragma omp parallel for num_threads(numThreads)  // 明确每列的独立性，确保线程间无共享变量冲突
		for (int j = 0; j < width; j++)
		{
			int feature3_start = -1;
			int roi_up = -1;
			int roi_down = -1;

			for (int i = 0; i < height - 3; i++)  //!? 与next_next_p的处理相联系
			{
				int p0 = binary[i * width + j];
				int p1 = binary[(i + 1) * width + j];
				int p2 = binary[(i + 2) * width + j];

				// 找到上边缘
				bool cond = (p0 > FEATURE3_CONTRAST_DEGREE && p1 > FEATURE3_CONTRAST_DEGREE && p2 > FEATURE3_CONTRAST_DEGREE);
				//!? 必须满足连续一段的像素均超过某个均值，避免椒盐噪声的影响！
				if (cond && feature3_start == -1) { feature3_start = i; }  // 记录上边界

				// 确定中心区域
				if (feature3_start != -1) {
					roi_up = feature3_start + roi_up_offset;
					roi_down = feature3_start + roi_down_offset;
				}

				// enhance operation
				binary[i * width + j] += (roi_down != -1 && i >= roi_up && i <= roi_down) ? 30 : 0;

				// 节约时间
				if (roi_down != -1 && i > roi_down) { break; }
			}
		}

		return 0;
	}



	int enhance_center_image_06122(
		unsigned char* binary, int width, int height, int numThreads)
	{
		int FEATURE3_CONTRAST_DEGREE = 50;
		const int roi_up_offset = FEATURE3_DIAMETER / 4;
		const int roi_down_offset = FEATURE3_DIAMETER / 4 * 3;
#pragma omp parallel for num_threads(4)
		for (int j = 0; j < width; j++)
		{
			int feature3_start = -1;
			int roi_up = -1;
			int roi_down = -1;

			for (int i = 0; i < height - 3; i++)  //!? 与next_next_p的处理相联系
			{
				int p0 = binary[i * width + j];
				int p1 = binary[(i + 1) * width + j];
				int p2 = binary[(i + 2) * width + j];

				// 找到上边缘
				bool cond = (p0 > FEATURE3_CONTRAST_DEGREE && p1 > FEATURE3_CONTRAST_DEGREE && p2 > FEATURE3_CONTRAST_DEGREE);
				//!? 必须满足连续一段的像素均超过某个均值，避免椒盐噪声的影响！
				if (cond && feature3_start == -1) { feature3_start = i; }  // 记录上边界

				// 确定中心区域
				if (feature3_start != -1) {
					roi_up = feature3_start + roi_up_offset;
					roi_down = feature3_start + roi_down_offset;

					for (int c = roi_up; c < roi_down + 1; c++)
					{
						binary[(c) * width + j] += 30;
					}

					break;
				}

			}
		}

		return 0;
	}

	void optimize_0609(cv::Mat& img1, cv::Mat& imgrst, BlurVersion2& blur2)
	{
		int _i = 0;
		cv::Mat _gray, _gray2;
		if (img1.channels() == 1)
		{
			cv::cvtColor(img1, imgrst, cv::COLOR_GRAY2BGR);
			img1.copyTo(_gray);
		}
		else
		{
			imgrst = img1.clone();
			cv::cvtColor(img1, _gray, cv::COLOR_BGR2GRAY);
		}

		/** block1 */
		//_gray2 = _gray.clone();
		//cv::blur(_gray, _gray2, cv::Size(21, 21));
		_gray2 = cv::Mat::zeros(_gray.size(), _gray.type());
		int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _gray2.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);
		//blur2.IM_BoxBlur_SSE_Blocks(_gray, _gray2, 10, 2, 2);


		/** block2 */
		_gray = _gray2 - _gray; // 提取高频信息

		enhance_center_image(_gray.ptr(0), _gray.cols, _gray.rows);


		/** block3 */
		int _iresult = picshadowx_ptr_0519(_gray.ptr(0), imgrst.ptr(0), 4, _gray.cols, _gray.rows);

		return;
	}


	void optimize_0612(cv::Mat& img1, cv::Mat& imgrst, BlurVersion2& blur2)
	{
		int _i = 0;
		cv::Mat _gray, _gray2;
		if (img1.channels() == 1)
		{
			cv::cvtColor(img1, imgrst, cv::COLOR_GRAY2BGR);
			img1.copyTo(_gray);
		}
		else
		{
			imgrst = img1.clone();
			cv::cvtColor(img1, _gray, cv::COLOR_BGR2GRAY);
		}

		/** block1 */
		//_gray2 = _gray.clone();
		//cv::blur(_gray, _gray2, cv::Size(21, 21));
		_gray2 = cv::Mat::zeros(_gray.size(), _gray.type());
		int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _gray2.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);
		//blur2.IM_BoxBlur_SSE_Blocks(_gray, _gray2, 10, 2, 2);


		/** block2 */
		_gray = _gray2 - _gray; // 提取高频信息

		enhance_center_image_06122(_gray.ptr(0), _gray.cols, _gray.rows, 4 );


		/** block3 */
		int _iresult = picshadowx_ptr_0519(_gray.ptr(0), imgrst.ptr(0), 4, _gray.cols, _gray.rows);

		return;
	}

	// enhance_center_image测试
	void experiment8(std::vector<cv::Mat> input) {
		auto start = std::chrono::high_resolution_clock::now();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		int total_pics_num = 1000;
		BlurVersion2 blur2 = BlurVersion2();


		/** standard process*/
		cv::Mat src1 = input[0];
		cv::Mat standard_gray, standard_gray2, standard_gray3;
		cv::Mat standard_imgrst = src1.clone();
		cv::cvtColor(src1, standard_gray, cv::COLOR_BGR2GRAY);  // 希望取消该方式!
		cv::blur(standard_gray, standard_gray2, cv::Size(21, 21));
		standard_gray3 = standard_gray2 - standard_gray;




		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			enhance_center_image_06122(standard_gray3.ptr(0), standard_gray3.cols, standard_gray3.rows, 4);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("enhance_center_image0612 {} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			enhance_center_image(standard_gray3.ptr(0), standard_gray3.cols, standard_gray3.rows);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("enhance_center_image0609 {} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));



		
		int standard_iresult = picshadowy(standard_gray3, &standard_imgrst, 4);



		/** experiments */
		//cv::Rect roi = cv::Rect(cv::Point(0, 1000), cv::Point(4095, 1200));
		//cv::Mat img1 = src1(roi);
		cv::Mat img1 = src1.clone();
		cv::Mat imgrst = img1.clone();
		optimize_0612(img1, imgrst, blur2);
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			optimize_0612(img1, imgrst, blur2);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("0612: {} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));


		cv::Mat img2 = src1.clone();
		cv::Mat imgrst2 = img1.clone();
		optimize_0609(img2, imgrst2, blur2);
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			optimize_0609(img2, imgrst2, blur2);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("0609: {} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

		cv::Mat img3 = src1.clone();
		cv::Mat imgrst3 = img1.clone();
		optimize_0612(img3, imgrst3, blur2);
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			optimize_0612(img3, imgrst3, blur2);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("0612: {} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

		cv::Mat img4 = src1.clone();
		cv::Mat imgrst4 = img1.clone();
		optimize_0609(img4, imgrst4, blur2);
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			optimize_0609(img4, imgrst4, blur2);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("0609: {} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));
	}

	// 测试int变量如果通过++越界，会变成什么
	// 结论：假如0~255； 则重新从0开始！
	void test1() {
		uint8_t count = 0;
		for (int i = 0; i < 512; i++) {
			count++;
			LOGD("count: {};", count);
		}
	}



	void A113_solver()
	{
		// https://github.com/BBuf/Image-processing-algorithm-Speed/blob/master/speed_rgb2gray_sse.cpp

		cv::Mat src1 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\1.jpg");
		cv::Mat src1200 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\1.jpg");
		//cv::Rect roi1(0, 1135, 4096, 200);
		//cv::Rect roi2(0, 1135, 4096, 400);

		cv::Rect roi1(0, 10, 10, 10);
		cv::Rect roi2(0, 10, 10, 10);

		cv::Mat local_image1 = src1200(roi1);  // 4096 * 200
		cv::Mat local_image2 = src1200(roi2);  // 4096 * 400
		cv::Mat local_image3;
		cv::hconcat(local_image1, local_image1, local_image3);  // 8192 * 200


		cv::Mat src2 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\1.jpg");
		cv::Mat src3 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\2.jpg");
		cv::Mat src4 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\3.jpg");
		cv::Mat src5 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\4.jpg");
		cv::Mat src6 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\5.jpg");
		cv::Mat src7 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\6.jpg");
		cv::Mat src8 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\7.jpg");
		cv::Mat src9 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\8.jpg");
		cv::Mat src10 = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\1111cmaketests\\9.jpg");

		std::vector<cv::Mat> input0 = { src1 };
		std::vector<cv::Mat> input1 = { src1, src2, };
		std::vector<cv::Mat> input2 = { src1, src2,src3,src4, };
		std::vector<cv::Mat> input3 = { src1, src2,src3,src4,src5,src6, };
		std::vector<cv::Mat> input4 = { src1, src2,src3,src4,src5,src6, src7,src8, };
		std::vector<cv::Mat> input5 = { src1, src2, src3,src4,src5,src6,src7,src8,src9, src10 };
		//std::vector<cv::Mat> input = { src1, src2};


		std::vector<cv::Mat> input01 = { local_image1 };
		std::vector<cv::Mat> input02 = { local_image2 };
		std::vector<cv::Mat> input03 = { local_image3 };

		//experiment4(input0);

		//studyBoxFilter();

		//LOGD("4096 * 200");
		//experiment6(input01);
		//experiment7(input01); 

		//LOGD("4096 * 400");
		//experiment6(input02);
		//experiment7(input02);

		//LOGD("8192 * 200");
		//experiment6(input03);
		//experiment7(input03);

		LOGD("720 * 168");
		experiment6(input0);
		experiment7(input0);
		experiment8(input0);





		//test1();


		return;
	}
}
