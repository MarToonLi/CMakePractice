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

using namespace std;
using namespace cv;

/*
功能：单元测试三行代码算法的效率


*/

#define Gaussian_Size 20
#define Gaussian_Size_2 (Gaussian_Size>>1)

#pragma execution_character_set("utf-8") 

namespace NA111 {
	int NUMTHREADS = 4;
	float Gaussian_Ker_XY[Gaussian_Size];


	class BlurVersion1
	{

	public:
		void CalcGaussCof(float Radius, float& B0, float& B1, float& B2, float& B3)
		{
			float Q, B;
			if (Radius >= 2.5)
				Q = (double)(0.98711 * Radius - 0.96330);                            //    对应论文公式11b
			else if ((Radius >= 0.5) && (Radius < 2.5))
				Q = (double)(3.97156 - 4.14554 * sqrt(1 - 0.26891 * Radius));
			else
				Q = (double)0.1147705018520355224609375;

			B = 1.57825 + 2.44413 * Q + 1.4281 * Q * Q + 0.422205 * Q * Q * Q;        //    对应论文公式8c
			B1 = 2.44413 * Q + 2.85619 * Q * Q + 1.26661 * Q * Q * Q;
			B2 = -1.4281 * Q * Q - 1.26661 * Q * Q * Q;
			B3 = 0.422205 * Q * Q * Q;

			B0 = 1.0 - (B1 + B2 + B3) / B;
			B1 = B1 / B;
			B2 = B2 / B;
			B3 = B3 / B;
		}

		void ConvertBGR8U2BGRAF(float* Src, float* Dest, int Width, int Height, int Stride)
		{
			//#pragma omp parallel for
			for (int Y = 0; Y < Height; Y++)
			{
				float* LinePS = Src + Y * Stride;
				float* LinePD = Dest + Y * Width * 3;
				for (int X = 0; X < Width; X++, LinePS += 3, LinePD += 3)
				{
					LinePD[0] = LinePS[0];    LinePD[1] = LinePS[1];    LinePD[2] = LinePS[2];
				}
			}
		}

		void ConvertBGR8U2BGRAF_SSE(float* Src, float* Dest, int Width, int Height, int Stride) {
			const int BlockSize = 4;
			int Block = (Width - 2) / BlockSize;
			__m128i Mask = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
			__m128i Zero = _mm_setzero_si128();
			for (int Y = 0; Y < Height; Y++) {
				float* LinePS = Src + Y * Stride;
				float* LinePD = Dest + Y * Width * 4;
				int X = 0;
				for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize * 3, LinePD += BlockSize * 4) {
					__m128i SrcV = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)LinePS), Mask);
					__m128i Src16L = _mm_unpacklo_epi8(SrcV, Zero);
					__m128i Src16H = _mm_unpackhi_epi8(SrcV, Zero);
					_mm_store_ps(LinePD + 0, _mm_cvtepi32_ps(_mm_unpacklo_epi16(Src16L, Zero)));
					_mm_store_ps(LinePD + 4, _mm_cvtepi32_ps(_mm_unpackhi_epi16(Src16L, Zero)));
					_mm_store_ps(LinePD + 8, _mm_cvtepi32_ps(_mm_unpacklo_epi16(Src16H, Zero)));
					_mm_store_ps(LinePD + 12, _mm_cvtepi32_ps(_mm_unpackhi_epi16(Src16H, Zero)));
				}
				for (; X < Width; X++, LinePS += 3, LinePD += 4) {
					LinePD[0] = LinePS[0];    LinePD[1] = LinePS[1];    LinePD[2] = LinePS[2];    LinePD[3] = 0;
				}
			}
		}

		void GaussBlurFromLeftToRight(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
		{
			//#pragma omp parallel for
			for (int Y = 0; Y < Height; Y++)
			{
				float* LinePD = Data + Y * Width * 3;
				//w[n-1], w[n-2], w[n-3]
				float BS1 = LinePD[0], BS2 = LinePD[0], BS3 = LinePD[0]; //边缘处使用重复像素的方案
				float GS1 = LinePD[1], GS2 = LinePD[1], GS3 = LinePD[1];
				float RS1 = LinePD[2], RS2 = LinePD[2], RS3 = LinePD[2];
				for (int X = 0; X < Width; X++, LinePD += 3)
				{
					LinePD[0] = LinePD[0] * B0 + BS1 * B1 + BS2 * B2 + BS3 * B3;
					LinePD[1] = LinePD[1] * B0 + GS1 * B1 + GS2 * B2 + GS3 * B3;         // 进行顺向迭代
					LinePD[2] = LinePD[2] * B0 + RS1 * B1 + RS2 * B2 + RS3 * B3;
					BS3 = BS2, BS2 = BS1, BS1 = LinePD[0];
					GS3 = GS2, GS2 = GS1, GS1 = LinePD[1];
					RS3 = RS2, RS2 = RS1, RS1 = LinePD[2];
				}
			}
		}

		void GaussBlurFromLeftToRight_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
			const __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
			const __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
			const __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
			const __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
			for (int Y = 0; Y < Height; Y++) {
				float* LinePD = Data + Y * Width * 4;
				__m128 V1 = _mm_set_ps(LinePD[3], LinePD[2], LinePD[1], LinePD[0]);
				__m128 V2 = V1, V3 = V1;
				for (int X = 0; X < Width; X++, LinePD += 4) {
					__m128 V0 = _mm_load_ps(LinePD);
					__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
					__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
					__m128 V = _mm_add_ps(V01, V23);
					V3 = V2; V2 = V1; V1 = V;
					_mm_store_ps(LinePD, V);
				}
			}
		}

		void GaussBlurFromRightToLeft(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
			for (int Y = 0; Y < Height; Y++) {
				//w[n+1], w[n+2], w[n+3]
				float* LinePD = Data + Y * Width * 3 + (Width * 3);
				float BS1 = LinePD[0], BS2 = LinePD[0], BS3 = LinePD[0]; //边缘处使用重复像素的方案
				float GS1 = LinePD[1], GS2 = LinePD[1], GS3 = LinePD[1];
				float RS1 = LinePD[2], RS2 = LinePD[2], RS3 = LinePD[2];
				for (int X = Width - 1; X >= 0; X--, LinePD -= 3)
				{
					LinePD[0] = LinePD[0] * B0 + BS3 * B1 + BS2 * B2 + BS1 * B3;
					LinePD[1] = LinePD[1] * B0 + GS3 * B1 + GS2 * B2 + GS1 * B3;         // 进行反向迭代
					LinePD[2] = LinePD[2] * B0 + RS3 * B1 + RS2 * B2 + RS1 * B3;
					BS1 = BS2, BS2 = BS3, BS3 = LinePD[0];
					GS1 = GS2, GS2 = GS3, GS3 = LinePD[1];
					RS1 = RS2, RS2 = RS3, RS3 = LinePD[2];
				}
			}
		}

		void GaussBlurFromRightToLeft_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
			const __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
			const __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
			const __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
			const __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
			for (int Y = 0; Y < Height; Y++) {
				float* LinePD = Data + Y * Width * 4 + (Width * 4);
				__m128 V1 = _mm_set_ps(LinePD[3], LinePD[2], LinePD[1], LinePD[0]);
				__m128 V2 = V1, V3 = V1;
				for (int X = Width - 1; X >= 0; X--, LinePD -= 4) {
					__m128 V0 = _mm_load_ps(LinePD);
					__m128 V03 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V3));
					__m128 V12 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V1));
					__m128 V = _mm_add_ps(V03, V12);
					V1 = V2; V2 = V3; V3 = V;
					_mm_store_ps(LinePD, V);
				}
			}
		}

		void GaussBlurFromTopToBottom(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
		{
			for (int Y = 0; Y < Height; Y++)
			{
				float* LinePD3 = Data + (Y + 0) * Width * 3;
				float* LinePD2 = Data + (Y + 1) * Width * 3;
				float* LinePD1 = Data + (Y + 2) * Width * 3;
				float* LinePD0 = Data + (Y + 3) * Width * 3;
				for (int X = 0; X < Width; X++, LinePD0 += 3, LinePD1 += 3, LinePD2 += 3, LinePD3 += 3)
				{
					LinePD0[0] = LinePD0[0] * B0 + LinePD1[0] * B1 + LinePD2[0] * B2 + LinePD3[0] * B3;
					LinePD0[1] = LinePD0[1] * B0 + LinePD1[1] * B1 + LinePD2[1] * B2 + LinePD3[1] * B3;
					LinePD0[2] = LinePD0[2] * B0 + LinePD1[2] * B1 + LinePD2[2] * B2 + LinePD3[2] * B3;
				}
			}
		}

		void GaussBlurFromTopToBottom_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
			const  __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
			const  __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
			const  __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
			const  __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
			for (int Y = 0; Y < Height; Y++)
			{
				float* LinePS3 = Data + (Y + 0) * Width * 4;
				float* LinePS2 = Data + (Y + 1) * Width * 4;
				float* LinePS1 = Data + (Y + 2) * Width * 4;
				float* LinePS0 = Data + (Y + 3) * Width * 4;
				for (int X = 0; X < Width * 4; X += 4)
				{
					__m128 V3 = _mm_load_ps(LinePS3 + X);
					__m128 V2 = _mm_load_ps(LinePS2 + X);
					__m128 V1 = _mm_load_ps(LinePS1 + X);
					__m128 V0 = _mm_load_ps(LinePS0 + X);
					__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
					__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
					_mm_store_ps(LinePS0 + X, _mm_add_ps(V01, V23));
				}
			}
		}

		void GaussBlurFromBottomToTop(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
			for (int Y = Height - 1; Y >= 0; Y--) {
				float* LinePD3 = Data + (Y + 3) * Width * 3;
				float* LinePD2 = Data + (Y + 2) * Width * 3;
				float* LinePD1 = Data + (Y + 1) * Width * 3;
				float* LinePD0 = Data + (Y + 0) * Width * 3;
				for (int X = 0; X < Width; X++, LinePD0 += 3, LinePD1 += 3, LinePD2 += 3, LinePD3 += 3) {
					LinePD0[0] = LinePD0[0] * B0 + LinePD1[0] * B1 + LinePD2[0] * B2 + LinePD3[0] * B3;
					LinePD0[1] = LinePD0[1] * B0 + LinePD1[1] * B1 + LinePD2[1] * B2 + LinePD3[1] * B3;
					LinePD0[2] = LinePD0[2] * B0 + LinePD1[2] * B1 + LinePD2[2] * B2 + LinePD3[2] * B3;
				}
			}
		}

		void GaussBlurFromBottomToTop_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
			const  __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
			const  __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
			const  __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
			const  __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
			for (int Y = Height - 1; Y >= 0; Y--) {
				float* LinePS3 = Data + (Y + 3) * Width * 4;
				float* LinePS2 = Data + (Y + 2) * Width * 4;
				float* LinePS1 = Data + (Y + 1) * Width * 4;
				float* LinePS0 = Data + (Y + 0) * Width * 4;
				for (int X = 0; X < Width * 4; X += 4) {
					__m128 V3 = _mm_load_ps(LinePS3 + X);
					__m128 V2 = _mm_load_ps(LinePS2 + X);
					__m128 V1 = _mm_load_ps(LinePS1 + X);
					__m128 V0 = _mm_load_ps(LinePS0 + X);
					__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
					__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
					_mm_store_ps(LinePS0 + X, _mm_add_ps(V01, V23));
				}
			}
		}

		void ConvertBGRAF2BGR8U(float* Src, float* Dest, int Width, int Height, int Stride)
		{
			//#pragma omp parallel for
			for (int Y = 0; Y < Height; Y++)
			{
				float* LinePS = Src + Y * Width * 3;
				float* LinePD = Dest + Y * Stride;
				for (int X = 0; X < Width; X++, LinePS += 3, LinePD += 3)
				{
					LinePD[0] = LinePS[0];    LinePD[1] = LinePS[1];    LinePD[2] = LinePS[2];
				}
			}
		}

		void ConvertBGRAF2BGR8U_SSE(float* Src, float* Dest, int Width, int Height, int Stride) {
			const int BlockSize = 4;
			int Block = (Width - 2) / BlockSize;
			//__m128i Mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
			__m128i MaskB = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
			__m128i MaskG = _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
			__m128i MaskR = _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
			__m128i Zero = _mm_setzero_si128();
			for (int Y = 0; Y < Height; Y++) {
				float* LinePS = Src + Y * Width * 4;
				float* LinePD = Dest + Y * Stride;
				int X = 0;
				for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize * 4, LinePD += BlockSize * 3) {
					__m128i SrcV = _mm_loadu_si128((const __m128i*)LinePS);
					__m128i B = _mm_shuffle_epi8(SrcV, MaskB);
					__m128i G = _mm_shuffle_epi8(SrcV, MaskG);
					__m128i R = _mm_shuffle_epi8(SrcV, MaskR);
					__m128i Ans1 = Zero, Ans2 = Zero, Ans3 = Zero;
					Ans1 = _mm_or_si128(Ans1, _mm_shuffle_epi8(B, _mm_setr_epi8(0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
					Ans1 = _mm_or_si128(Ans1, _mm_shuffle_epi8(G, _mm_setr_epi8(-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
					Ans1 = _mm_or_si128(Ans1, _mm_shuffle_epi8(R, _mm_setr_epi8(-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));

					Ans2 = _mm_or_si128(Ans2, _mm_shuffle_epi8(B, _mm_setr_epi8(-1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
					Ans2 = _mm_or_si128(Ans2, _mm_shuffle_epi8(G, _mm_setr_epi8(1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
					Ans2 = _mm_or_si128(Ans2, _mm_shuffle_epi8(R, _mm_setr_epi8(-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));

					Ans3 = _mm_or_si128(Ans3, _mm_shuffle_epi8(B, _mm_setr_epi8(-1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
					Ans3 = _mm_or_si128(Ans3, _mm_shuffle_epi8(G, _mm_setr_epi8(-1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
					Ans3 = _mm_or_si128(Ans3, _mm_shuffle_epi8(R, _mm_setr_epi8(2, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));

					_mm_storeu_si128((__m128i*)(LinePD + 0), Ans1);
					_mm_storeu_si128((__m128i*)(LinePD + 4), Ans2);
					_mm_storeu_si128((__m128i*)(LinePD + 8), Ans3);
				}
				for (; X < Width; X++, LinePS += 4, LinePD += 3) {
					LinePD[0] = LinePS[0]; LinePD[1] = LinePS[1]; LinePD[2] = LinePS[2];
				}
			}
		}

		void GaussBlur(float* Src, float* Dest, int Width, int Height, int Stride, float Radius)
		{
			float B0, B1, B2, B3;
			float* Buffer = (float*)malloc(Width * (Height + 6) * sizeof(float) * 3);
			CalcGaussCof(Radius, B0, B1, B2, B3);
			ConvertBGR8U2BGRAF(Src, Buffer + 3 * Width * 3, Width, Height, Stride);
			GaussBlurFromLeftToRight(Buffer + 3 * Width * 3, Width, Height, B0, B1, B2, B3);
			GaussBlurFromRightToLeft(Buffer + 3 * Width * 3, Width, Height, B0, B1, B2, B3);        //    如果启用多线程，建议把这个函数写到GaussBlurFromLeftToRight的for X循环里，因为这样就可以减少线程并发时的阻力

			memcpy(Buffer + 0 * Width * 3, Buffer + 3 * Width * 3, Width * 3 * sizeof(float));
			memcpy(Buffer + 1 * Width * 3, Buffer + 3 * Width * 3, Width * 3 * sizeof(float));
			memcpy(Buffer + 2 * Width * 3, Buffer + 3 * Width * 3, Width * 3 * sizeof(float));

			GaussBlurFromTopToBottom(Buffer, Width, Height, B0, B1, B2, B3);

			memcpy(Buffer + (Height + 3) * Width * 3, Buffer + (Height + 2) * Width * 3, Width * 3 * sizeof(float));
			memcpy(Buffer + (Height + 4) * Width * 3, Buffer + (Height + 2) * Width * 3, Width * 3 * sizeof(float));
			memcpy(Buffer + (Height + 5) * Width * 3, Buffer + (Height + 2) * Width * 3, Width * 3 * sizeof(float));

			GaussBlurFromBottomToTop(Buffer, Width, Height, B0, B1, B2, B3);

			ConvertBGRAF2BGR8U(Buffer + 3 * Width * 3, Dest, Width, Height, Stride);

			free(Buffer);
		}

		void GaussBlur_SSE(float* Src, float* Dest, int Width, int Height, int Stride, float Radius)
		{
			float B0, B1, B2, B3;
			float* Buffer = (float*)_mm_malloc(Width * (Height + 6) * sizeof(float) * 4, 16);
			CalcGaussCof(Radius, B0, B1, B2, B3);
			ConvertBGR8U2BGRAF_SSE(Src, Buffer + 3 * Width * 4, Width, Height, Stride);
			GaussBlurFromLeftToRight_SSE(Buffer + 3 * Width * 4, Width, Height, B0, B1, B2, B3);        //    在SSE版本中，这两个函数占用的时间比下面两个要多,不过C语言版本也是一样的
			GaussBlurFromRightToLeft_SSE(Buffer + 3 * Width * 4, Width, Height, B0, B1, B2, B3);        //    如果启用多线程，建议把这个函数写到GaussBlurFromLeftToRight的for X循环里，因为这样就可以减少线程并发时的阻力

			memcpy(Buffer + 0 * Width * 4, Buffer + 3 * Width * 4, Width * 4 * sizeof(float));
			memcpy(Buffer + 1 * Width * 4, Buffer + 3 * Width * 4, Width * 4 * sizeof(float));
			memcpy(Buffer + 2 * Width * 4, Buffer + 3 * Width * 4, Width * 4 * sizeof(float));

			GaussBlurFromTopToBottom_SSE(Buffer, Width, Height, B0, B1, B2, B3);

			memcpy(Buffer + (Height + 3) * Width * 4, Buffer + (Height + 2) * Width * 4, Width * 4 * sizeof(float));
			memcpy(Buffer + (Height + 4) * Width * 4, Buffer + (Height + 2) * Width * 4, Width * 4 * sizeof(float));
			memcpy(Buffer + (Height + 5) * Width * 4, Buffer + (Height + 2) * Width * 4, Width * 4 * sizeof(float));

			GaussBlurFromBottomToTop_SSE(Buffer, Width, Height, B0, B1, B2, B3);

			ConvertBGRAF2BGR8U_SSE(Buffer + 3 * Width * 4, Dest, Width, Height, Stride);

			_mm_free(Buffer);
		}
	};

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
			for (int X = 0; X < Block * BlockSize; X += BlockSize)
			{
				__m128i SrcV1 = _mm_loadu_si128((__m128i*)(Array + Radius + Radius - X - 3));
				__m128i SrcV2 = _mm_loadu_si128((__m128i*)(Array + Radius + Length - X - 5));
				_mm_storeu_si128((__m128i*)(Array + X), _mm_shuffle_epi32(SrcV1, _MM_SHUFFLE(0, 1, 2, 3)));
				_mm_storeu_si128((__m128i*)(Array + Radius + Length + X), _mm_shuffle_epi32(SrcV2, _MM_SHUFFLE(0, 1, 2, 3)));
			}
			//　　处理剩余不能被SSE优化的数据
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
			for (; X < Length; X += BlockSize)
			{
				Sum1 = _mm_add_epi32(Sum1, _mm_loadu_si128((__m128i*)(Array + X + 0)));
				Sum2 = _mm_add_epi32(Sum2, _mm_loadu_si128((__m128i*)(Array + X + 4)));
			}
			// 水平相加Sum1和Sum2的32位整型
			__m128i SumTotal = _mm_add_epi32(Sum1, Sum2);

			// 将128位寄存器拆分为高位和低位相加
			SumTotal = _mm_add_epi32(SumTotal, _mm_srli_si128(SumTotal, 8));
			SumTotal = _mm_add_epi32(SumTotal, _mm_srli_si128(SumTotal, 4));

			// 获取最终标量结果
			int Sum = _mm_extract_epi32(SumTotal, 0);


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
				if (ColValue != NULL)     free(ColValue);
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
	};


	// ===============================================================================================

	// ParallelAdd 面向picshadowx第一部分
	class ParallelAdd : public cv::ParallelLoopBody  //参考官方给出的answer，构造一个并行的循环体类
	{
	public:
		ParallelAdd(const cv::Mat& _src, int* _blackcout2, int* _total2)
		{
			src = _src;
			blackcout = _blackcout2;
			total2 = _total2;

			src_data = src.ptr<uchar>(0);       // 数据起始指针
			step = src.step;                             //获取每一行的元素总个数（相当于cols*channels，等同于step1)
		}

		void operator()(const cv::Range& range) const             //重载操作符（）
		{
			for (int col = range.start; col < range.end; col++)
			{
				//#pragma omp parallel for num_threads(2)
				for (int row = 0; row < src.rows; row++)
				{
					if (*(src_data + row * step + col) > 50) {
						blackcout[col]++;                  //垂直投影按列在x轴进行投影
						++*total2;
					}
				}
			}
		}

	private:
		cv::Mat src;
		mutable int* total2;
		int* blackcout;

		const uchar* src_data;
		int step;
	};

	// ParallelAdd 面向picshadowx第三部分
	class ParallelShow1 : public cv::ParallelLoopBody//参考官方给出的answer，构造一个并行的循环体类
	{
	public:
		ParallelShow1(const cv::Mat& _src, cv::Mat* _show, int* _blackcout2)
		{
			src = _src;
			blackcout = _blackcout2;
			show = _show;

			src_data = src.ptr<uchar>(0);       // 数据起始指针
			step = src.step;                             //获取每一行的元素总个数（相当于cols*channels，等同于step1)
		}


		void operator()(const cv::Range& range) const             //重载操作符（）
		{
			for (int col = range.start; col < range.end; col++)
			{
				for (int j = 0; j < blackcout[col]; j++)
				{
					show->at<cv::Vec3b>(show->rows - 1 - j, col)[0] = 255; //翻转到下面，便于观看
					show->at<cv::Vec3b>(show->rows - 1 - j, col)[1] = 255; //翻转到下面，便于观看
					show->at<cv::Vec3b>(show->rows - 1 - j, col)[2] = 0;   //翻转到下面，便于观看
				}
			}
		}

	private:
		cv::Mat src;
		cv::Mat* show;
		int* blackcout;

		const uchar* src_data;
		int step;
	};

	// ParallelAdd 面向picshadowx第四部分
	class ParallelShow2 : public cv::ParallelLoopBody//参考官方给出的answer，构造一个并行的循环体类
	{
	public:
		ParallelShow2(const cv::Mat& _src, cv::Mat* _show, int* _blackcout2, int _avg)
		{
			src = _src;
			blackcout = _blackcout2;
			show = _show;
			avg = _avg;

			src_data = src.ptr<uchar>(0);       // 数据起始指针
			step = src.step;                             //获取每一行的元素总个数（相当于cols*channels，等同于step1)
		}


		void operator()(const cv::Range& range) const             //重载操作符（）
		{
			for (int col = range.start; col < range.end; col++)
			{
				if (blackcout[col] > avg + 5.3)
				{
					for (int j = 0; j < src.rows / 10; j++)
					{
						show->at<cv::Vec3b>(j, col)[0] = 128;//翻转到下面，便于观看
						show->at<cv::Vec3b>(j, col)[1] = 128;//翻转到下面，便于观看
						show->at<cv::Vec3b>(j, col)[2] = 128;//翻转到下面，便于观看
					}
				}
			}
		}

	private:
		cv::Mat src;
		cv::Mat* show;
		int* blackcout;
		int avg;

		const uchar* src_data;
		int step;
	};

	// ===========================================================================================================
	
	// picshadowx 第一部分
	void block1(cv::Mat binary, int* blackcout, int& _total, double& _avg) {
		int numThreads = NA111::NUMTHREADS;

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
		_avg = _total * 1.0 / binary.cols;

		return;
	}

	// parallel_for_
	void block1_2(cv::Mat binary, int* blackcout, int& _total, double& _avg) {
		int numThreads = NA111::NUMTHREADS;

		int* blackcout2 = new int[binary.cols];
		memset(blackcout2, 0, binary.cols * 4);
		int val = 0;
		int* _total2 = &val;

		ParallelAdd parallelAdd(binary, blackcout2, _total2);
		cv::parallel_for_(cv::Range(0, binary.cols), parallelAdd, 25);  //隐式调用

		return;
	}

	// picshadowx 第二部分
	void block2(cv::Mat& binary, int* blackcout, int& _total, double& _avg, int &_res)
	{
		int numThreads = NA111::NUMTHREADS;

		#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < binary.cols; i++)
		{
			if (blackcout[i] > _avg + 5.3)
			{
				_res = 1;
				break;
			}
		}
	}

	// picshadowx 第三部分
	void block3(cv::Mat& binary, int* blackcout, int& _total, double& _avg, int& _res, cv::Mat* show)
	{
		int numThreads = NA111::NUMTHREADS;

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
	}

	// picshadowx 第四部分
	void block4(cv::Mat& binary, int* blackcout, int& _total, double& _avg, int& _res, cv::Mat* show)
	{
		int numThreads = NA111::NUMTHREADS;

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

	// optimize: cv::Vec3b
	void block4_2(cv::Mat& binary, int* blackcout, int& _total, double& _avg, int& _res, cv::Mat* show)
	{
		int numThreads = NA111::NUMTHREADS;
		cv::Vec3b pixel = cv::Vec3b(255, 255, 0);

		#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < binary.cols; i++)
		{
			int count = blackcout[i];
			for (int j = 0; j < count; j ++)
			{
				show->at<cv::Vec3b>(show->rows - 1 - j, i) = pixel;
			}
		}
	}
	
	// optimize: cv::Vec3b + 4路循环展开
	void block4_3(cv::Mat& binary, int* blackcout, int& _total, double& _avg, int& _res, cv::Mat* show)
	{
		int numThreads = NA111::NUMTHREADS;
		cv::Vec3b pixel = cv::Vec3b(255, 255, 0);

#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < binary.cols; i++)
		{
			int count = blackcout[i];

			int j = 0;
			for (; j < count; j += 4)
			{
				show->at<cv::Vec3b>(show->rows - 1 - j, i) = pixel;
				show->at<cv::Vec3b>(show->rows - 1 - j - 1, i) = pixel;
				show->at<cv::Vec3b>(show->rows - 1 - j - 2, i) = pixel;
				show->at<cv::Vec3b>(show->rows - 1 - j - 3, i) = pixel;
			}

			for (; j < count; j++)
			{
				show->at<cv::Vec3b>(show->rows - 1 - j, i) = pixel;
			}
		}
	}
	
	
	// ========================================================

	// test: picshadowx with loops
	int picshadowx_test(cv::Mat binary, cv::Mat* show)
	{
		int _res = 2;
		int* blackcout = new int[binary.cols];
		memset(blackcout, 0, binary.cols * 4);
		int _total = 0;
		double _avg;


		int* blackcout2 = new int[binary.cols];
		memset(blackcout2, 0, binary.cols * 4);
		int _total2 = 0;
		double _avg2;

		int for_count1 = 10000;
		int for_count2 = 1;
		int for_count3 = 1;
		int for_count4 = 0;

		//int numThreads = omp_get_max_threads();
		int numThreads = NA111::NUMTHREADS;

		auto t1 = std::chrono::high_resolution_clock::now();
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);


		std::this_thread::sleep_for(std::chrono::seconds(3));

		t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < for_count1; i++)
		{
			block1_2(binary, blackcout2, _total2, _avg2);
		}
		t2 = std::chrono::high_resolution_clock::now();
		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		LOGD("block1_2: {};", duration1);


		std::this_thread::sleep_for(std::chrono::seconds(3));

		t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < for_count1; i++)
		{
			block1(binary, blackcout, _total, _avg);
		}
		t2 = std::chrono::high_resolution_clock::now();
		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		LOGD("block1: {};", duration1);


		t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < for_count2; i++)
		{
			block2(binary, blackcout, _total, _avg, _res);
		}
		t2 = std::chrono::high_resolution_clock::now();
		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		LOGD("block2: {};", duration1);


		t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < for_count3; i++)
		{
			block3(binary, blackcout, _total, _avg, _res, show);
		}
		t2 = std::chrono::high_resolution_clock::now();
		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		LOGD("block3: {};", duration1);


		t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < for_count4; i++)
		{
			block4_3(binary, blackcout, _total, _avg, _res, show);
		}
		t2 = std::chrono::high_resolution_clock::now();
		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		LOGD("block4_3: {};", duration1);


		t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < for_count4; i++)
		{
			block4_2(binary, blackcout, _total, _avg, _res, show);
		}
		t2 = std::chrono::high_resolution_clock::now();
		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

		LOGD("block4_2: {};", duration1);

		delete[] blackcout;
		blackcout = nullptr;

		return _res;
	}

	int picshadowx(cv::Mat binary, cv::Mat* show)
	{
		int _res = 2;
		int* blackcout = new int[binary.cols];
		memset(blackcout, 0, binary.cols * 4);
		int _total = 0;

		//int numThreads = omp_get_max_threads();
		int numThreads = NA111::NUMTHREADS;

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

	// 仅对doing进行omp加速
	int doing_omp1(cv::Mat binary)
	{
		try
		{
			LOGD("[single] start.");
			doing(binary);   // its purpose: 1. warm up; 2. the result of data will be passed to software for normally run. 
			LOGD("[single] end.");


			int numThreads = 20;
			LOGD("numThreads: {};", numThreads);

			int total_pics_num = 1400 * 20;
			LOGD("[batch1] start.");
			auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(numThreads)
			for (int i = 0; i < total_pics_num; i++)
			{
				cv::Mat data_copy = binary.clone();      // data need to be deep copy! 
				doing(data_copy);               // 析构的时候，发现有6个引用，所以崩溃……
			}
			auto end = std::chrono::high_resolution_clock::now();
			LOGD("[batch] end.");
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

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


		return -1;
	}

	// 测试 高斯模糊
	void doing(cv::Mat imgori) {
		int _i = 0;
		cv::Mat lastimg = imgori.clone();
		cv::Mat imgrst;
		cv::Mat _gray, _gray2, _gray3;
		if (lastimg.channels() == 1)
		{
			cv::cvtColor(lastimg, imgrst, cv::COLOR_GRAY2BGR);
			lastimg.copyTo(_gray);
		}
		else
		{
			imgrst = lastimg.clone();
			cv::cvtColor(imgori, _gray, cv::COLOR_BGR2GRAY);
		}

		cv::Mat _grayA1, _grayA2;
		auto t11 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
		{
			cv::blur(_gray, _grayA1, cv::Size(21, 21));  //! 均值提取低频信息
		}
		auto t12 = std::chrono::high_resolution_clock::now();
		_grayA2 = _grayA1 - _gray;                       //! 实现对高频信息的提取（背景去除）


		cv::Mat _grayB1, _grayB2;
		cv::Mat _grayC1, _grayC2;
		_grayB1 = _gray.clone();
		_grayC1 = _gray.clone();
		lastimg.convertTo(_gray3, CV_32F);
		BlurVersion1 blur1 = BlurVersion1();
		BlurVersion2 blur2 = BlurVersion2();
		auto t31 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
		{
			// 效果与原装不一样！
			//int result1 = blur2.IM_BoxBlur_C(_gray.ptr<uchar>(0), _grayB1.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);
			int result2 = blur2.IM_BoxBlur_SSE(_gray.ptr<uchar>(0), _grayC1.ptr<uchar>(0), _gray.cols, _gray.rows, _gray.cols, 10);
			//blur1.GaussBlur_SSE(_gray3.ptr<float>(0), _gray2.ptr<float>(0), _gray.cols, _gray.rows, _gray.cols * 3, 20);
			//LOGD("s");
		}
		auto t32 = std::chrono::high_resolution_clock::now();
		_grayB2 = _grayB1 - _gray;

		for (int i = 0; i < 1000; i++)
		{
			int _iresult = picshadowx(_gray, &imgrst);
		}
		auto t4 = std::chrono::high_resolution_clock::now();

		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11);
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(t32 - t31);


		LOGD("durations1: {}; duration2: {};", duration1, duration2);


		int* a = new int[10];
		int* b = new int[10];

		unsigned char a1[] = { 0,1,2,3,4,5,6,7,8,9 };
		unsigned char b1[] = { 9,10,2,3,4,5,6,7,8,9 };



	}

	// 测试 picshadowx_test
	void test_doing(cv::Mat imgori) {
		int _i = 0;
		cv::Mat lastimg = imgori.clone();
		cv::Mat imgrst;
		cv::Mat _gray, _gray2, _gray3;
		if (lastimg.channels() == 1)
		{
			cv::cvtColor(lastimg, imgrst, cv::COLOR_GRAY2BGR);
			lastimg.copyTo(_gray);
		}
		else
		{
			imgrst = lastimg.clone();
			cv::cvtColor(imgori, _gray, cv::COLOR_BGR2GRAY);
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++)
		{
			cv::blur(_gray, _gray2, cv::Size(20, 20));
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++)
		{
			_gray3 = _gray2 - _gray;
		}
		_gray = _gray2 - _gray;

		auto t3 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++)
		{
			//int _iresult = picshadowx(_gray, &imgrst);
			int _iresult = picshadowx_test(_gray, &imgrst);
		}
		auto t4 = std::chrono::high_resolution_clock::now();

		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
		auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
		LOGD("duration1: {}; duration2: {}; duration3: {};", duration1, duration2, duration3);
	}

    void A111_solver()
    {
		// https://github.com/BBuf/Image-processing-algorithm-Speed/blob/master/speed_rgb2gray_sse.cpp

        cv::Mat src = cv::imread("D:\\Myself\\MachineVision\\resources\\GuangXian\\test_1562__ORI_DA2710107.jpg");


		int total_pics_num = 1;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < total_pics_num; i++) {
			doing(src);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("{} pics time: {}; single pic time: {};", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));

        return;
    }
}
