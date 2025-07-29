#pragma once
#include "slogger.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>
#include <omp.h>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/utils/logger.hpp>  // 在加载模型前设置日志级别

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include "scommon.h"

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include "logging.h"
#include "cuda_runtime_api.h"


#include <cstdlib>
#include<numeric>

using namespace nvinfer1;
using namespace plugin;

namespace Anomalib 
{



    #define CHECK(call)                                   \
    do                                                    \
    {                                                     \
        const cudaError_t error_code = call;              \
        if (error_code != cudaSuccess)                    \
        {                                                 \
            printf("CUDA Error:\n");                      \
            printf("    File:       %s\n", __FILE__);     \
            printf("    Line:       %d\n", __LINE__);     \
            printf("    Error code: %d\n", error_code);   \
            printf("    Error text: %s\n",                \
                cudaGetErrorString(error_code));          \
            exit(1);                                      \
        }                                                 \
    } while (0)




    // 自定义配置结构
    struct Configuration
    {
    public:
        float objThreshold;    //Object Confidence threshold对象置信度阈值
        float nmsThreshold;    // Non-maximum suppression threshold非最大抑制阈值
        std::vector<float> confThresholds;   // Confidence threshold置信度阈值
        int bboxAreaThreshold; //去除小检测框面积阈值；
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



    const static char* kInputTensorName = "images";
    const static char* kOutputTensorName = "output";

    const static int kNumClass = 80;
    const static int kBatchSize = 1;

    const static int kInputH = 640;
    const static int kInputW = 640;

    const static float kNmsThresh = 0.7f;
    const static float kConfThresh = 0.5f;

    const static int kMaxInputImageSize = 3000 * 3000;
    const static int kMaxNumOutputBbox = 1000;

    struct alignas(float) Detection {   //!? float类型内存对齐说明符，显式指定结构体的内存对齐方式，能提升性能！
        float bbox[4];
        float conf;
        float class_id;
    };

    struct Anomalib_Detection {
        float pred_score;
        float pred_label;
        cv::Mat pred_mask;
        cv::Mat anomaly_map;
    };


    struct AffineMatrix {
        float value[6];
    };


    // 声明模板
    template<typename T>
    bool BlobFromImage(std::vector<cv::Mat>& imgs, T& iBlob);



    class Onnx_Anomalib
    {
    public:
        Onnx_Anomalib(Configuration config);

        void normalize_(std::vector<cv::Mat>& frames);		// 归一化函数
        void draw(cv::Mat& imgrst, Anomalib_Detection& output);
        cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left, cv::Scalar& add_color);
        bool onnx_detect(std::vector<cv::Mat>& frames, std::vector<Anomalib_Detection>& output);

    private:
        std::vector<cv::Mat> dstImgs;
        std::vector<float> confThresholds;
        float nmsThreshold;
        float objThreshold;
        int bboxAreaThreshold;
        int inpWidth;
        int inpHeight;
        int nout;
        int num_proposal;
        int num_classes;
        const bool keep_ratio = false;
        std::string classes[2] = { "fire", "Smog" };

        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1");  // 初始化和管理 ONNX Runtime 的环境。它控制日志记录、模型加载和执行
        // ORT_LOGGING_LEVEL_ERROR 表示只会记录错误级别及以上的日志，这意味着只有发生错误时才会输出日志信息。
        // "yolov5-6.1": ONNX Runtime 环境的名称, 以便在日志中标识。
        Ort::Session* ort_session;    // 初始化Session指针选项
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();  //初始化Session对象

        std::vector<float> input_image_;		// 输入图片
        std::vector<char*> input_names;  // 定义一个字符指针vector
        std::vector<char*> output_names; // 定义一个字符指针vector
        std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs  ，二维vector
        std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++标准


    };



}