#pragma once
#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"
#include "macros.h"
class Yolov5TRTContext {

public:
    float* data;
    float* prob;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void* buffers[2];
    cudaStream_t stream;
    int inputIndex;
    int outputIndex;
};
extern "C" API void* Init(char* model_path);
extern "C" API  void Detect(void* h, int rows, int cols, unsigned char* src_data, float(*res_array)[6]);
extern "C" API  void cuda_free(void* h);
   