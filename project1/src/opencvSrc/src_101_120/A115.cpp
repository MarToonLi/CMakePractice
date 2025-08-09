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
#include <typeinfo>

using namespace std;
using namespace cv;
using namespace ml;

#pragma execution_character_set("utf-8") 

/**********************************
C++中Opencv的传统机器学习算法
************************************/




namespace NA115 {

	/***
	Case1: SVM运用
	***/


	namespace Case1 {

		typedef struct HOG_PARAM
		{
			Size winsize;
			Size blocksize;
			Size blockstep;
			Size cellsize;
			int bins;

		}hogParam;

		bool getFilesOfDir(const char* path, std::vector<std::string>& files, const char* imgType)
		{
			string cpath = path, ctype = imgType;
			cpath = cpath + "//*" + ctype;

			intptr_t  hFile = 0;
			struct _finddata_t fileinfo;
			files.clear();
			if ((hFile = _findfirst(cpath.c_str(), &fileinfo)) != -1)
			{
				do
				{
					if (!(fileinfo.attrib & _A_SUBDIR))
						files.push_back(fileinfo.name);
				} while (_findnext(hFile, &fileinfo) == 0);
				_findclose(hFile);
				return true;
			}
			else
				return false;
		}

		void getDirsOfDir(const char* path, std::vector<std::string>& files)
		{
			string cpath = path;

			//文件句柄 
			long  hFile = 0;
			//文件信息 
			struct _finddata_t fileinfo;
			string p;
			if ((hFile = (long)_findfirst(p.assign(cpath).append("\\*").c_str(), &fileinfo)) != -1)
			{
				do
				{
					if ((fileinfo.attrib & _A_SUBDIR))
					{
						if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
						{
							files.push_back(fileinfo.name);
							//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
						}

					}
				} while (_findnext(hFile, &fileinfo) == 0);
				_findclose(hFile);
			}
		}

		void generateSamples()
		{
			//string folderPath = "D:\\ProjectData\\motor_Data_1\\20210520\\OK";
			string folderPath = "C:\\Users\\vcl-robot\\Desktop\\德宏";
			vector<string> imgPaths;
			getFilesOfDir(folderPath.c_str(), imgPaths, ".bmp");

			int nImgs = (int)imgPaths.size();
			for (int i = 0; i < nImgs; i++)
			{
				cout << i << "/" << nImgs << endl;

				string srcPath = folderPath + "\\" + imgPaths[i];
				//string dstpath_ori = folderPath + "\\samples\\" + imgPaths[i].substr(0, imgPaths[i].size() - 4) + "-";
				string dstpath_ori = folderPath + "\\samples\\" + to_string(i) + "-";

				Mat src = imread(srcPath.c_str(), 1);

				if (src.empty())
					continue;

				//Rect roi(1192, 762, 180, 180);
				Rect roi(884, 710, 180, 180);
				int dstNo = 0;
				int xOffset = 10, yOffset = 10, xStep = 2, yStep = 2;
				for (int x = roi.x - xOffset; x <= roi.x + xOffset; x += xStep)
				{
					for (int y = roi.y - yOffset; y <= roi.y + yOffset; y += yStep)
					{
						Rect curRoi = Rect(x, y, roi.width, roi.height);

						if (curRoi.x < 0 || curRoi.y < 0 || curRoi.x + curRoi.width >= src.cols || curRoi.y + curRoi.height >= src.rows)
							continue;

						Mat curImg = src(curRoi).clone();

						dstNo++;
						string dstpath = dstpath_ori + to_string(dstNo) + ".png";
						imwrite(dstpath.c_str(), curImg);

						//flip(curImg, curImg, 1); 
						//dstNo++;
						//dstpath = dstpath_ori + to_string(dstNo) + ".png";
						//imwrite(dstpath.c_str(), curImg);

						//flip(curImg, curImg, 0); 
						//dstNo++;
						//dstpath = dstpath_ori + to_string(dstNo) + ".png";
						//imwrite(dstpath.c_str(), curImg);

						//flip(curImg, curImg, 1); 
						//dstNo++;
						//dstpath = dstpath_ori + to_string(dstNo) + ".png";
						//imwrite(dstpath.c_str(), curImg);
					}
				}
			}

		}

		void generateSamplesPosTrans()
		{
			string folderPath = "ofData\\NG-data\\0-ori";
			string dstfolderPath = "ofData\\NG-data\\1-gen";
			vector<string> imgPaths;
			getFilesOfDir(folderPath.c_str(), imgPaths, ".bmp");

			int nImgs = (int)imgPaths.size();
			for (int i = 0; i < nImgs; i++)
			{
				cout << i << "/" << nImgs << endl;

				string srcPath = folderPath + "\\" + imgPaths[i];
				Mat src = imread(srcPath.c_str(), 1);
				if (src.empty())	continue;
				string dstpath;

				int dstNo = 0;
				Mat curImg = src.clone();
				dstNo++;
				dstpath = dstfolderPath + "\\" + imgPaths[i].substr(0, imgPaths[i].size() - 4) + to_string(dstNo) + ".png";
				imwrite(dstpath.c_str(), curImg);

				flip(curImg, curImg, 1);
				dstNo++;
				dstpath = dstfolderPath + "\\" + imgPaths[i].substr(0, imgPaths[i].size() - 4) + to_string(dstNo) + ".png";
				imwrite(dstpath.c_str(), curImg);

				flip(curImg, curImg, 0);
				dstNo++;
				dstpath = dstfolderPath + "\\" + imgPaths[i].substr(0, imgPaths[i].size() - 4) + to_string(dstNo) + ".png";
				imwrite(dstpath.c_str(), curImg);

				flip(curImg, curImg, 1);
				dstNo++;
				dstpath = dstfolderPath + "\\" + imgPaths[i].substr(0, imgPaths[i].size() - 4) + to_string(dstNo) + ".png";
				imwrite(dstpath.c_str(), curImg);
			}

		}

		void ofSVM_Train_HOG(Ptr<SVM> svm, hogParam hParam, string pos_dir, string neg_dir, string svm_path)
		{
			vector<string> train_pos_paths;
			vector<string> train_neg_paths;
			getFilesOfDir(pos_dir.c_str(), train_pos_paths, ".png");
			getFilesOfDir(neg_dir.c_str(), train_neg_paths, ".png");
			int nPos = (int)train_pos_paths.size();
			int nNeg = (int)train_neg_paths.size();
			vector<string> imgPath;
			vector<int> imgCatg;
			for (int i = 0; i < nPos; i++)
			{
				string cImgPath = pos_dir + "\\" + train_pos_paths[i];
				imgPath.push_back(cImgPath);
				imgCatg.push_back(1);
			}
			for (int i = 0; i < nNeg; i++)
			{
				string cImgPath = neg_dir + "\\" + train_neg_paths[i];
				imgPath.push_back(cImgPath);
				imgCatg.push_back(0);
			}

			Mat trainData;
			Mat trainLabel;
			trainLabel = Mat::zeros((int)imgPath.size(), 1, CV_32SC1);
			Mat src, trainImg;
			for (int i = 0; i < nPos + nNeg; i++)
			{
				src = imread(imgPath[i].c_str(), 1);
				resize(src, trainImg, hParam.winsize, 0, 0, INTER_AREA);

				HOGDescriptor* hog = new HOGDescriptor(hParam.winsize, hParam.blocksize, hParam.blockstep, hParam.cellsize, hParam.bins);

				vector<float>descriptors;
				hog->compute(trainImg, descriptors);
				delete hog;

				if (i == 0)
					trainData = Mat::zeros((int)imgPath.size(), (int)descriptors.size(), CV_32FC1);

				int n = 0;
				for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
				{
					trainData.at<float>(i, n) = *iter;
					n++;
				}

				trainLabel.at<int>(i, 0) = imgCatg[i];
			}

			svm->train(trainData, ROW_SAMPLE, trainLabel);  //训练分类器
			std::cout << "Finishing training..." << endl;

			//将训练好的SVM模型保存为xml文件
			svm->SVM::save(svm_path.c_str());
		}

		void ofSVM_Test_HOG(Ptr<SVM> svm, hogParam hParam, string test_dir, string test_res, vector<int>& vres, int& nPos, int& nNeg)
		{
			vector<string> test_paths;
			getFilesOfDir(test_dir.c_str(), test_paths, ".png");

			int nTest = (int)test_paths.size();
			Mat src, trainImg;
			nPos = 0;
			nNeg = 0;
			clock_t ts, te;
			double timeall(0);

			fstream prdRes;	prdRes.open(test_res.c_str(), ofstream::out);

			for (int i = 0; i < nTest; i++)
			{
				string cImgPath = test_dir + "\\" + test_paths[i];
				src = imread(cImgPath.c_str(), 1);


				ts = clock();
				resize(src, trainImg, hParam.winsize, 0, 0, INTER_AREA);

				HOGDescriptor* hog = new HOGDescriptor(hParam.winsize, hParam.blocksize, hParam.blockstep, hParam.cellsize, hParam.bins);
				vector<float>descriptors;
				hog->compute(trainImg, descriptors);
				delete hog;

				//ts = clock();
				int cres = (int)svm->predict(descriptors);
				te = clock();
				timeall += (te - ts);

				vres.push_back(cres);
				prdRes << test_paths[i] << " --- Category:" << std::to_string(cres) << endl;

				if (cres == 1) nPos++;
				else nNeg++;
			}


			double timeavg = timeall * 1.0 / nTest;
			cout << "pre-time: " << timeavg << "ms" << endl;

			prdRes << endl;
			prdRes << "Predicted Pos Samples: " << std::to_string(nPos) << endl;
			prdRes << "Predicted Neg Samples: " << std::to_string(nNeg) << endl;
			prdRes << "Average Processing Time: " << std::to_string(timeavg) << " ms" << endl;
			prdRes.close();
		}

		void ofSVM_HOG()
		{
			hogParam hParam;
			//hParam.winsize = Size(64, 128);
			//hParam.blocksize = Size(16, 16);
			//hParam.blockstep = Size(8, 8);
			//hParam.cellsize = Size(8, 8);
			//hParam.bins = 9;


			hParam.winsize = Size(32, 8);
			hParam.blocksize = Size(4, 4);
			hParam.blockstep = Size(4, 4);
			hParam.cellsize = Size(4, 4);
			hParam.bins = 9;

			//hParam.winsize = Size(64, 64);
			//hParam.blocksize = Size(32, 32);
			//hParam.blockstep = Size(4, 4);
			//hParam.cellsize = Size(8, 8);
			//hParam.bins = 9;


			string root_path = "F:\\Projects\\CMakePractice\\resources\\";

			string train_pos_dir = root_path + "ofData\\trainData\\pos";
			string train_neg_dir = root_path + "ofData\\trainData\\neg";
			string of_svm_path = root_path + "ofData\\of_svm_inty.xml";
			string test_dir1 = root_path + "ofData\\testData\\pos";
			string test_res1 = root_path + "ofData\\testData\\pos_res.txt";
			string test_dir2 = root_path + "ofData\\testData\\neg";
			string test_res2 = root_path + "ofData\\testData\\neg_res.txt";

			Ptr<SVM> svm = SVM::create();//SVM分类器
			svm->setType(SVM::C_SVC);
			svm->setC(0.01);
			svm->setKernel(SVM::LINEAR);
			svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

			cout << "training..." << endl;
			ofSVM_Train_HOG(svm, hParam, train_pos_dir, train_neg_dir, of_svm_path);
			//svm = SVM::load(lexy_svm_path.c_str());

			if (!svm)
			{
				cout << "Load file failed..." << endl;
				return;
			}

			cout << "testing..." << endl;
			vector<int> vResDir1, vResDir2, vResDir3, vResDir4, vResDir5;
			int nPos1, nNeg1, nPos2, nPos4, nPos5, nNeg2, nPos3, nNeg3, nNeg4, nNeg5;
			ofSVM_Test_HOG(svm, hParam, test_dir1, test_res1, vResDir1, nPos1, nNeg1);
			ofSVM_Test_HOG(svm, hParam, test_dir2, test_res2, vResDir2, nPos2, nNeg2);

			cout << "test1: pos-" << nPos1 << ", neg-" << nNeg1 << endl;
			cout << "test2: pos-" << nPos2 << ", neg-" << nNeg2 << endl;

		}

		void ofSVM1()
		{
			Size winsize(64, 128), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
			int bins(9);

			string train_pos_dir = "D:\\ProjectData\\lexy_Data\\20201201\\彩色\\samples\\gray\\train\\pos";
			string train_neg_dir = "D:\\ProjectData\\lexy_Data\\20201201\\彩色\\samples\\gray\\train\\neg";
			string lexy_svm_path = "lexy_svm_hog.xml";

			string test_dir1 = "D:\\ProjectData\\lexy_Data\\20201201\\彩色\\samples\\gray\\pos";
			string test_dir2 = "D:\\ProjectData\\lexy_Data\\20201201\\彩色\\samples\\gray\\neg";

			vector<string> train_pos_paths;
			vector<string> train_neg_paths;
			getFilesOfDir(train_pos_dir.c_str(), train_pos_paths, ".png");
			getFilesOfDir(train_neg_dir.c_str(), train_neg_paths, ".png");
			int nPos = (int)train_pos_paths.size();
			int nNeg = (int)train_neg_paths.size();
			vector<string> imgPath;
			vector<int> imgCatg;
			for (int i = 0; i < nPos; i++)
			{
				string cImgPath = train_pos_dir + "\\" + train_pos_paths[i];
				imgPath.push_back(cImgPath);
				imgCatg.push_back(1);
			}
			for (int i = 0; i < nNeg; i++)
			{
				string cImgPath = train_neg_dir + "\\" + train_neg_paths[i];
				imgPath.push_back(cImgPath);
				imgCatg.push_back(0);
			}

			Mat trainData;
			Mat trainLabel;
			trainLabel = Mat::zeros((int)imgPath.size(), 1, CV_32FC1);
			Mat src, trainImg;
			for (int i = 0; i < nPos + nNeg; i++)
			{
				src = imread(imgPath[i].c_str(), 0);
				resize(src, trainImg, winsize, 0, 0, INTER_AREA);

				HOGDescriptor* hog = new HOGDescriptor(winsize, blocksize, blockstep, cellsize, bins);

				vector<float>descriptors;
				hog->compute(trainImg, descriptors);
				delete hog;

				if (i == 0)
					trainData = Mat::zeros((int)imgPath.size(), (int)descriptors.size(), CV_32FC1);

				int n = 0;
				for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
				{
					trainData.at<float>(i, n) = *iter;
					n++;
				}

				trainLabel.at<float>(i, 0) = (float)imgCatg[i];
			}

			Ptr<SVM> svm = SVM::create();//SVM分类器
			svm->setType(SVM::C_SVC);
			svm->setC(0.01);
			svm->setKernel(SVM::LINEAR);
			svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));
			std::cout << "Starting training..." << endl;

			svm->train(trainData, ROW_SAMPLE, trainLabel);//训练分类器
			std::cout << "Finishing training..." << endl;

			//将训练好的SVM模型保存为xml文件
			svm->SVM::save(lexy_svm_path.c_str());

			svm->load(lexy_svm_path.c_str());



		}

		void ofSVM_Train_Inty(Ptr<SVM> svm, int cHeight, int cWidth, string pos_dir, string neg_dir, string svm_path)
		{
			vector<string> train_pos_paths;
			vector<string> train_neg_paths;
			getFilesOfDir(pos_dir.c_str(), train_pos_paths, ".png");
			getFilesOfDir(neg_dir.c_str(), train_neg_paths, ".png");
			int nPos = (int)train_pos_paths.size();
			int nNeg = (int)train_neg_paths.size();
			vector<string> imgPath;
			vector<int> imgCatg;
			for (int i = 0; i < nPos; i++)
			{
				string cImgPath = pos_dir + "\\" + train_pos_paths[i];
				imgPath.push_back(cImgPath);
				imgCatg.push_back(1);
			}
			for (int i = 0; i < nNeg; i++)
			{
				string cImgPath = neg_dir + "\\" + train_neg_paths[i];
				imgPath.push_back(cImgPath);
				imgCatg.push_back(0);
			}

			Mat trainData;
			Mat trainLabel;
			trainLabel = Mat::zeros((int)imgPath.size(), 1, CV_32SC1);
			Mat src, trainImg, trainImg0;
			for (int i = 0; i < nPos + nNeg; i++)
			{
				src = imread(imgPath[i].c_str(), 0);
				resize(src, trainImg0, Size(cWidth, cHeight), 0, 0, INTER_LINEAR);

				cv::normalize(trainImg0, trainImg, 1, 0, NORM_MINMAX, CV_32FC1);

				vector<float>descriptors(trainImg.cols);

				float* blackcout = new float[trainImg.cols];
				memset(blackcout, 0, trainImg.cols * 4);

				for (int i = 0; i < trainImg.rows; i++)
				{
					for (int j = 0; j < trainImg.cols; j++)
					{
						blackcout[j] += 1 - trainImg.at<float>(i, j);// *1.0 / 255;
					}
				}

				for (int j = 0; j < trainImg.cols; j++)
				{
					descriptors[j] = blackcout[j] / cHeight;
				}

				if (i == 0)
					trainData = Mat::zeros((int)imgPath.size(), (int)descriptors.size(), CV_32FC1);

				int n = 0;
				for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
				{
					trainData.at<float>(i, n) = *iter;
					n++;
				}

				trainLabel.at<int>(i, 0) = imgCatg[i];
			}

			svm->train(trainData, ROW_SAMPLE, trainLabel);//训练分类器
			std::cout << "Finishing training..." << endl;

			//将训练好的SVM模型保存为xml文件
			svm->SVM::save(svm_path.c_str());
		}

		void ofSVM_Test_Inty(Ptr<SVM> svm, int cHeight, int cWidth, string test_dir, string test_res, vector<int>& vres, int& nPos, int& nNeg)
		{
			vector<string> test_paths;
			getFilesOfDir(test_dir.c_str(), test_paths, ".png");

			int nTest = (int)test_paths.size();
			Mat src, testImg, testImg0;
			nPos = 0;
			nNeg = 0;
			clock_t ts, te;
			double timeall(0);

			fstream prdRes;	prdRes.open(test_res.c_str(), ofstream::out);

			for (int i = 0; i < nTest; i++)
			{
				string cImgPath = test_dir + "\\" + test_paths[i];
				src = imread(cImgPath.c_str(), 0);


				ts = clock();


				resize(src, testImg0, Size(cWidth, cHeight), 0, 0, INTER_LINEAR);


				cv::normalize(testImg0, testImg, 1, 0, NORM_MINMAX, CV_32FC1);

				vector<float>descriptors(testImg.cols);

				float* blackcout = new float[testImg.cols];
				memset(blackcout, 0, testImg.cols * 4);

				for (int i = 0; i < testImg.rows; i++)
				{
					for (int j = 0; j < testImg.cols; j++)
					{
						blackcout[j] += 1 - testImg.at<float>(i, j);// *1.0 / 255;
					}
				}

				for (int j = 0; j < testImg.cols; j++)
				{
					descriptors[j] = blackcout[j] / cHeight;
				}





				//ts = clock();
				int cres = (int)svm->predict(descriptors);
				te = clock();
				timeall += (te - ts);

				vres.push_back(cres);
				prdRes << test_paths[i] << " --- Category:" << std::to_string(cres) << endl;

				if (cres == 1) nPos++;
				else nNeg++;
			}


			double timeavg = timeall * 1.0 / nTest;
			cout << "pre-time: " << timeavg << "ms" << endl;

			prdRes << endl;
			prdRes << "Predicted Pos Samples: " << std::to_string(nPos) << endl;
			prdRes << "Predicted Neg Samples: " << std::to_string(nNeg) << endl;
			prdRes << "Average Processing Time: " << std::to_string(timeavg) << " ms" << endl;
			prdRes.close();
		}

		void ofSVM_Inty()
		{

			int nHeight = 168, nWidth = 360;

			string root_path = "F:\\Projects\\CMakePractice\\resources\\";

			string train_pos_dir = root_path + "ofData\\trainData\\pos";
			string train_neg_dir = root_path + "ofData\\trainData\\neg";
			string of_svm_path = root_path + "ofData\\of_svm_inty.xml";
			string test_dir1 = root_path + "ofData\\testData\\pos";
			string test_res1 = root_path + "ofData\\testData\\pos_res.txt";
			string test_dir2 = root_path + "ofData\\testData\\neg";
			string test_res2 = root_path + "ofData\\testData\\neg_res.txt";

			Ptr<SVM> svm = SVM::create();//SVM分类器
			svm->setType(SVM::C_SVC);
			svm->setC(0.01);
			svm->setKernel(SVM::LINEAR);

			svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

			cout << "training..." << endl;
			ofSVM_Train_Inty(svm, nHeight, nWidth, train_pos_dir, train_neg_dir, of_svm_path);
			//svm = SVM::load(lexy_svm_path.c_str());

			if (!svm)
			{
				cout << "Load file failed..." << endl;
				return;
			}

			cout << "testing..." << endl;
			vector<int> vResDir1, vResDir2, vResDir3, vResDir4, vResDir5;
			int nPos1, nNeg1, nPos2, nPos4, nPos5, nNeg2, nPos3, nNeg3, nNeg4, nNeg5;
			ofSVM_Test_Inty(svm, nHeight, nWidth, test_dir1, test_res1, vResDir1, nPos1, nNeg1);
			ofSVM_Test_Inty(svm, nHeight, nWidth, test_dir2, test_res2, vResDir2, nPos2, nNeg2);

			cout << "test1: pos-" << nPos1 << ", neg-" << nNeg1 << endl;
			cout << "test2: pos-" << nPos2 << ", neg-" << nNeg2 << endl;

		}
	
	
	}

	void A115_solver()
	{
		LOGD("ss");

		Case1::ofSVM_HOG();

		return;
	}
}
