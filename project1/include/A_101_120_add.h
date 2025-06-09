#include<opencv2/highgui.hpp>
#include<opencv2/ml/ml.hpp>  

#include<windows.h>
#include<iostream>
#include<map>
#include<string>
#include<time.h>
#include<time.h>
#include<io.h>
#include<direct.h>
#include<fstream>


#ifndef INCREMENTAL_MEAN_H
#define INCREMENTAL_MEAN_H

class IncrementalMean {
private:
    double sum;
    unsigned long count;

public:
    IncrementalMean();  // 构造函数声明
    void addValue(double value);  // 添加新值
    double getMean() const;  // 获取当前均值
    void reset();  // 重置计算
};


using namespace std;
using namespace cv;
using namespace ml;

void imgDeburring(cv::Mat src, cv::Mat& dst, int uThres, int vThres, cv::Size size, int dThres);
void imgAnalyze(Mat src, Mat& enSrc);
void copperAnalyze(Mat src, vector<Rect>& vRoi, Mat& srcInfo);

#endif // INCREMENTAL_MEAN_H