#pragma once
#include "logger.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <omp.h>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/imgproc/types_c.h>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>



// 自定义配置结构
struct Configuration
{
public:
    int bboxAreaThreshold; //去除小检测框面积阈值；
    float confThreshold; // Confidence threshold置信度阈值
    float nmsThreshold;  // Non-maximum suppression threshold非最大抑制阈值
    float objThreshold;  //Object Confidence threshold对象置信度阈值
    std::string modelpath;
};



// 定义BoxInfo结构类型
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;



class YOLOv5
{
public:
    YOLOv5(Configuration config);
    bool detect(std::vector<cv::Mat>& frames, std::vector<std::vector<BoxInfo>>& output);
private:
    std::vector<cv::Mat> dstImgs;
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    int bboxAreaThreshold;
    int inpWidth;
    int inpHeight;
    int nout;
    int num_proposal;
    int num_classes;
    std::string classes[2] = { "fire", "Smog"};
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1");

    const bool keep_ratio = true;
    std::vector<float> input_image_;		// 输入图片
    void normalize_(std::vector<cv::Mat>& frames);		// 归一化函数
    void nms(std::vector<BoxInfo>& input_boxes);
    cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);

   
    Ort::Session* ort_session;    // 初始化Session指针选项
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();  //初始化Session对象
    //SessionOptions sessionOptions;
    std::vector<char*> input_names;  // 定义一个字符指针vector
    std::vector<char*> output_names; // 定义一个字符指针vector
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs  ，二维vector
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++标准
};




