#ifndef TYPEC_OCR_H
#define TYPEC_OCR_H

#include <fstream>
#include <io.h>
#include <string>
//#include <opencv.hpp>
#include <opencv2/opencv.hpp>  //!? point to pay attention to when updating algos.
#include "../utils/logger.h"
#include <vector>


#include <onnxruntime_cxx_api.h>
//#include <postprocess_op.h>
//#include <preprocess_op.h>

#include "typec_ocr_prepostprocess.h"

//using namespace std;
//using namespace cv;
//using namespace Ort;


struct Configuration
{
public:
    int bboxAreaThreshold; //去除小检测框面积阈值；
    float confThreshold; // Confidence threshold置信度阈值
    float nmsThreshold;  // Non-maximum suppression threshold非最大抑制阈值
    float objThreshold;  //Object Confidence threshold对象置信度阈值
    std::string modelpath;
};


struct OCRConfiguration
{
    float plug_length_up = 1600;     //plug长度阈值
    float plug_length_down = 1400;   //plug长度阈值
    float plug_ratio_length = 1.f;   //ratio mm/pixel


    float ratio_location = 1.0f; //ratio mm/pixel
    float ratio_location_back = 1.f;


    float brightness_up = 200;      //正面灰度阈值
    float brightness_down = 65;     //正面灰度阈值
    float character_height_up = 200;//正面高度阈值
    float character_height_down = 0;//正面高度阈值


    float back_brightness_up = 255;      //反面灰度阈值
    float back_brightness_down = 65;     //反面灰度阈值
    float back_character_height_up = 250;//反面高度阈值
    float back_character_height_down = 0;//反面高度阈值


    float quality_score = 0.3f;
    float quality_score_i = 0.15f;
    float quality_score_l = 0.15f;
    float quality_score_t = 0.3f;
    float avg_score_thld_ = 0.6f;
    float quality_score_banxing = 0.1f;


    float quality_score_back = 0.3f;
    float quality_score_i_back = 0.15f;
    float quality_score_l_back = 0.15f;
    float quality_score_t_back = 0.3f;
    float avg_score_thld_back = 0.5f;
    float quality_score_banxing_back = 0.1f;

    float rotate_thld_value = 100;
};



struct CheckParam
{
    std::string cam_serial;

    //! 命名规则：
    //! 缺陷名称_阈值参数名称；
    //! 缺陷名称_算法参数名称；

    //! DetNet_config
    int use_gpu_det = 1;
    int use_dilation = 0;
    float det_db_thresh = 0.3;
    float det_db_box_thresh = 0.5;
    float det_db_unclip_ratio = 2.0;


    //! DetNet_config_back
    int use_gpu_det_back = 1;
    int use_dilation_back = 0;
    float det_db_thresh_back = 0.3;
    float det_db_box_thresh_back = 0.5;
    float det_db_unclip_ratio_back = 2.0;



    //! rec_config
    int use_gpu_rec = 1;


    //! rec_config_back
    int use_gpu_rec_back = 1;




    //! custom_param
    // front面 字符位置偏差
    float left_length_low = 500;
    float left_length_high = 625;
    float right_length_low = 500;
    float right_length_high = 670;
    float first_line_height_low = 200;
    float first_line_height_high = 300;
    float second_line_height_low = 50;
    float second_line_height_high = 150;




    //! plug_measure_config
    int thld_value = 100;
    int lt_x = -200;
    int lt_y = -140;
    int lt_w = 150;
    int lt_h = 193;
    int rt_x = 1460;
    int rt_y = -156;
    int rt_w = 150;
    int rt_h = 217;
    int lb_x = -388;
    int lb_y = 1394;
    int lb_w = 200;
    int lb_h = 233;
    int rb_x = 1712;
    int rb_y = 1372;
    int rb_w = 80;
    int rb_h = 181;
    int lm_x = -300;
    int lm_y = 148;
    int lm_w = 163;
    int lm_h = 1053;
    int rm_x = 1626;
    int rm_y = 166;
    int rm_w = 139;
    int rm_h = 1033;




    // segAlgParam
    float convertScaleParam = 1.1;
    float convertScaleDeltaParam = 30;
    float adaptiveThldBlockSize = 55;
    float adaptiveThldSubtractValue = 35;
    float backfaceGrayThld = 250;
    float erodeEleSize = 3;
    float areaForFilter1 = 100;
    float areaForFilter2 = 100;




    // custom_param_back
    float height_low = 70;
    float height_high = 200;




    // config
    float plug_length_up = 1600;
    float plug_length_down = 1400;
    float plug_ratio_length = 1.f;

    float ratio_location = 1.0f;
    float ratio_location_back = 1.f;

    float brightness_up = 200;
    float brightness_down = 65;
    float back_brightness_up = 255;
    float back_brightness_down = 65;

    float character_height_up = 200;
    float character_height_down = 0;
    float back_character_height_up = 250;
    float back_character_height_down = 0;

    float quality_score = 0.35f;
    float quality_score_i = 0.15f;
    float quality_score_l = 0.15f;
    float quality_score_t = 0.3f;
    float avg_score_thld_ = 0.61f;
    float quality_score_banxing = 0.1f;

    float quality_score_back = 0.3f;
    float quality_score_i_back = 0.15f;
    float quality_score_l_back = 0.15f;
    float quality_score_t_back = 0.3f;
    float avg_score_thld_back = 0.5f;
    float quality_score_banxing_back = 0.1f;

    float rotate_thld_value = 100.0f;



    // Param_ModelConfig
    float ironcladScratchThreshold;
    float multiIroncladScratchThreshold;
    float ironcladShinyMarkThreshold;
    float th_conf;
    float th_nms;
    float th_obj;
};



struct DetNet_config
{
	std::string modelpath_det;
    double det_db_thresh;
    double det_db_box_thresh;
    double det_db_unclip_ratio;
    bool use_dilation;
    bool use_gpu_det;
};

struct RecNet_config
{
	std::string modelpath_rec;
	std::string dictFile;

    bool use_gpu_rec;
};

struct CharactDefectNet_config
{
	std::string modelpath;
	std::string classesFile;
    bool use_gpu_defect;
};

enum TypeOcrStatus
{
    OK,                   // 0 
    NG_CHARACTER_DEFECT,  //1
    NG_CHARACTER_COLOR,   //2
    NG_CHARACTER_SIZE,    //3
    NG_PLUG_LENGTH,       //4
    NG_LOCATION,          //5
    NG_SN_COMPARE_ERROR,  //6
    NG_DEFECT_SEGMENT,    //7
    NG_DEFECT_AVERAGE,    //8
    NG_CHARACTER_NUM,     //9
    NG_LINES_NUM,         //10
    NG_NULL,              //11
    WR_SN_UNREAD_APP,     //12
    NG_UNDEFINED,         //13
    NG_ironcladScratch,
    NG_ironcladShinyMark,
    NG_COMPANY_CODE_ERROR,
};



#define CASE_STR(x) case x: return #x; break;

string print_enum_EN(TypeOcrStatus e);






struct TypeOcrRes
{
	std::vector<TypeOcrStatus> status;
	std::vector<std::string> ocr_content;
};



struct OCRErrorValue
{
    std::vector<std::vector<double>> single_score_error;
    std::vector<double> average_score_error;
    std::vector<double> average_color_error;
};

class OcrPredictor
{
public:
	OcrPredictor(DetNet_config det_config, RecNet_config rec_config);
	~OcrPredictor();

	void run(cv::Mat src, int input_face, std::vector<std::tuple<std::string, std::vector<std::vector<int>>>> &ocr_results);
    void get_input_face(cv::Mat src, int& input_face);
private:
    //Det
    bool use_gpu_det_ = false;
    int gpu_id_det_ = 0;
    int gpu_mem_det_ = 4000;
    int cpu_math_library_num_threads_det_ = 4;
    bool use_mkldnn_det_ = false;

    std::string limit_type_ = "max";
    int limit_side_len_ = 1600;//960,1280//1600//1920

    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    std::string det_db_score_mode_ = "slow";
    bool use_dilation_ = false;

    bool visualize_ = true;
    bool use_tensorrt_ = false;
    std::string precision_ = "fp32";

    std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
    std::vector<float> scale_ = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
    bool is_scale_ = true;
    float ratio_h{};
    float ratio_w{};

    // pre-process
    PaddleOCR::ResizeImgType0 resize_op_;
    PaddleOCR::Normalize normalize_op_;
    PaddleOCR::Permute permute_op_;

    //det onnx
    std::string detModelPath;
    Ort::Session* net = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DET");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    std::vector<char*> input_names;
    std::vector<char*> output_names;

    //Rec
    bool use_gpu_rec_ = false;
    int gpu_id_rec_ = 0;
    int gpu_mem_rec_ = 4000;
    int cpu_math_library_num_threads_rec_ = 4;
    bool use_mkldnn_rec_ = false;

    std::vector<float> mean_rec_ = { 0.5f, 0.5f, 0.5f };
    std::vector<float> scale_rec_ = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
    bool is_scale_rec_ = true;
    bool use_tensorrt_rec_ = false;
    std::string precision_rec_ = "fp32";
    int rec_batch_num_ = 6;
    int rec_img_h_ = 32; //v2:32 --- v3/v4:48
    int rec_img_w_ = 320;
    std::vector<int> rec_image_shape_ = { 3, rec_img_h_, rec_img_w_ };
    // pre-process
    PaddleOCR::CrnnResizeImg resize_op_rec_;
    PaddleOCR::Normalize normalize_op_rec_;
    PaddleOCR::PermuteBatch permute_op_rec_batch_;
    PaddleOCR::Permute permute_op_rec_;

    PaddleOCR::Resize resize_op_rec_v3_;

    std::vector<std::string> alphabet;

    //onnx
    std::string recModelPath;
    std::string recDictPath;
    Ort::Env env_rec = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "REC");
    Ort::Session* ort_session_rec = nullptr;
    Ort::SessionOptions sessionOptions_rec = Ort::SessionOptions();
    std::vector<char*> input_names_rec;
    std::vector<char*> output_names_rec;
    std::vector<std::vector<int64_t>> input_node_dims_rec; // >=1 outputs
    std::vector<std::vector<int64_t>> output_node_dims_rec; // >=1 outputs
};

class CharacterDefectPredictor
{
public:
	CharacterDefectPredictor(CharactDefectNet_config charact_config);
	~CharacterDefectPredictor();

    void run(cv::Mat src, cv::Mat templ, int input_face, float& cls_idx);

private:
    size_t w = 224;
    size_t h = 224;

    std::vector<float> mean_defect_ = { 0.485f, 0.456f, 0.406f };
    std::vector<float> scale_defect_ = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
    bool is_scale_defect_ = true;

    PaddleOCR::Normalize normalize_op_defect_;
    PaddleOCR::Permute permute_op_defect_;

    //onnx
    std::string defectModelPath;
    std::string defectClassPath;

    bool use_gpu_defect_ = false;

    Ort::Env env_defect = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DEFECT");
    Ort::Session* ort_session_defect = nullptr;
    Ort::SessionOptions sessionOptions_defect = Ort::SessionOptions();

    std::vector<char*> input_names_defect;
    std::vector<char*> output_names_defect;

    std::vector<double> softmax(const std::vector<double>& z);

    std::vector<std::string> labels;
};

struct singleSegALgParam
{
    float convertScaleParam; //对比度增强的scale参数
    float convertScaleDeltaParam;//对比度增强的灰度偏差值
    float adaptiveThldBlockSize;//自适应二值化块大小
    float adaptiveThldSubtractValue;//自适应二值化相减值
    float backfaceGrayThld;//反面的二值化阈值
    float erodeEleSize;//腐蚀操作的核大小
    float areaForFilter1;//过滤噪声的面积大小
    float areaForFilter2;//单字符二次分割的过滤面积大小
};

class CharacterDefectDetector
{
public:
    CharacterDefectDetector(std::string model_path_list, std::string cls_file_path, bool use_gpu_defect, singleSegALgParam segAlgParam);
    ~CharacterDefectDetector();

    void run(cv::Mat src, int input_face, float quality_score, float quality_i, float quality_l, float quality_t, float avg_score_thld_, float quality_score_banxing, std::vector<std::tuple<std::string, std::vector<std::vector<int>>>> ocr_results, TypeOcrRes& res, cv::Mat& dst, OCRErrorValue &err_value, std::string sn_code);
    //bool sort_by_x(tuple<int, int> a, tuple<int, int> b);
    void run_color_defect(cv::Mat src, int input_face, int brightness_up, int brightness_down, std::vector<std::tuple<std::string, std::vector<std::vector<int>>>> ocr_results, TypeOcrRes& res, cv::Mat& dst, OCRErrorValue& err_value);
    void run_character_size_defect(cv::Mat src, float character_height_up, float character_height_down, int is_front, singleSegALgParam segAlgParam, std::vector<std::tuple<std::string, std::vector<std::vector<int>>>> ocr_results, TypeOcrRes& res, cv::Mat& dst, OCRErrorValue& err_value);

    std::map<char, CharacterDefectPredictor*> ocr_charact_detect_list;
    std::map<std::string, cv::Mat> ocr_charact_hal_model_list;
private:
    std::string cls_model_path_list;
    std::string cls_path;
    bool use_gpu_defect_d = false;
    singleSegALgParam segAlgParam_;
    
public:
    void seg_single_character(cv::Mat roi, cv::Point left_top, int input_face, singleSegALgParam segAlgParam, std::vector<cv::Rect>& single_character);
    void seg_single_character_by_temp(cv::Mat roi, cv::Point left_top, int is_front, int i_line, std::string text, std::vector<cv::Rect>& single_character);
    std::vector<cv::Rect> rect_split_by_characters_by_temp(cv::Mat roi, int is_front, int i_line, std::string character_text, std::vector<cv::Rect> vec_rects, int width_thld);
    bool rect_split_by_characters_by_temp_judge(cv::Mat roi, int is_front, int i_line, std::string character_text, std::vector<cv::Rect> vec_rects, int width_thld, std::vector<cv::Rect> vec_rects_out);
};

struct PlugMeasureConfig
{
    int thld_value;

    cv::Rect rect_left_top;
    cv::Rect rect_right_top;
    cv::Rect rect_left_bottom;
    cv::Rect rect_right_bottom;
    cv::Rect rect_left_middle;
    cv::Rect rect_right_middle;
};

class PlugMeasurement
{
public:
    PlugMeasurement(PlugMeasureConfig plug_measure_config);
    ~PlugMeasurement();
    void run_measure(cv::Mat src, float length_up, float length_down, float ratio, std::vector<float>& line_output, TypeOcrRes& res, cv::Mat& dst);

private:
    int thld_value_ = 100;

    cv::Rect rect_left_top_;
    cv::Rect rect_right_top_;
    cv::Rect rect_left_bottom_;
    cv::Rect rect_right_bottom_;
    cv::Rect rect_left_middle_;
    cv::Rect rect_right_middle_;
};

struct CharactLocCustomParam
{
    float left_length_high;
    float left_length_low;
    float right_length_high;
    float right_length_low;
    float first_line_height_high;
    float first_line_height_low;
    float second_line_height_high;
    float second_line_height_low;
};
struct CharactLocCustomParamBack
{
    float height_high;
    float height_low;
};
class CharacterLocation
{
public:
    CharacterLocation();
    ~CharacterLocation();

    void run_location(cv::Mat src, CharactLocCustomParam custom_param, float ratio, std::vector<float> infer_line, std::vector<std::tuple<std::string, std::vector<std::vector<int>>>> ocr_results, TypeOcrRes& res, cv::Mat& dst);

private:

};

class CharacterLocationBack
{
public:
    CharacterLocationBack();
    ~CharacterLocationBack();

    void run_location(cv::Mat src, CharactLocCustomParamBack custom_param, float ratio, std::vector<float> infer_line, std::vector<std::tuple<std::string, std::vector<std::vector<int>>>> ocr_results, TypeOcrRes& res, cv::Mat& dst);

private:

};

void rotate_img_before_ocr(cv::Mat src, int thld_value_, cv::Mat& rot_src, cv::Mat& imgrst, cv::Rect &mask);
void get_input_face_new(cv::Mat src, int& input_face);

#endif //TYPEC_OCR_H
