#define USE_FP16
//#define USE_INT8

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output0";

const static int kNumClass = 2;
const static int kBatchSize = 1;

const static int kInputH = 1120;
const static int kInputW = 1120;

const static float kNmsThresh = 0.7f;
const static float kConfThresh = 0.5f;

const static int kMaxInputImageSize = 3000*3000;
const static int kMaxNumOutputBbox = 1000;

struct alignas(float) Detection {
  float bbox[4];
  float conf;
  float class_id;
};
