#ifndef PUTTEXT_H_
#define PUTTEXT_H_


#include "ibasealgorithm.h"
#include "algosettingdlg.h"
#include "yaml-cpp/yaml.h"
#include <onnxruntime_cxx_api.h>
#include "typec_ocr.h"
#include "intsafe.h"
#include <iostream>
#include <filesystem>
#include <windows.h>
#include <string>
#include "yolov5.hpp"
#include "common.h"
#include "icroland-scratch-grade.h"
#include <ft2build.h> 
#include <freetype/freetype.h>
#include FT_FREETYPE_H

#include "CvxText.h"



namespace wikky_algo
{

#define OK                              0x00100000
#define ERROR_CHARACTER_DEFECT          0x00110001
#define ERROR_CHARACTER_COLOR           0x00110002
#define ERROR_CHARACTER_SIZE            0x00110003
#define ERROR_CHARACTER_LOCATION        0x00110004
#define ERROR_CHARACTER_LENGTH          0x00110005
#define ERROR_PLUG_LENGTH               0x00120001
#define ERROR_NULL                      0x00109000
#define STR(x) #x

    // 前四位，位置错误大类；
//NG_character_error

    QByteArray print_enum(TypeOcrStatus e);

    std::string vectors2string(std::vector<TypeOcrStatus> vectors);
    std::string vectors2string(std::vector<BoxInfo> vectors);

    std::string addressToString(void* address);

    bool outputGPU();


    class Alg_Foundation::Impl : public wikky_algo::IBaseAlg
    {
    private:
        std::thread::id tid;
        _Thrd_t t;
        char* buf = new char[10];
        cv::Mat lastimg;
        std::shared_ptr<Qtalgosettingdlg> algosettingdlg = nullptr;
        std::string m_scamserial;
        cv::Mat* imgrst_ptr = nullptr;

        double SMALL_AREA_THRESHOLD = 2.3e+06;
        double LARGE_AREA_THRESHOLD = 2.7e+06;

        // checkparam
        CheckParam m_checkparam;
        YAML::Node m_yamlparams;
        void updateparamfromdlg(CheckParam _param);
        double calBoxArea(BoxInfo boxinfo);
        double calBoxDiagonalLength(BoxInfo boxinfo);
        bool defectAccumJudge(const cv::Mat& image, const std::vector<std::vector<BoxInfo>> output, ErrorStatus NG_STATUS, bool& isdefect, float accumStandardVal, cv::Mat main_area_mask);

    /// <summary>

        OcrPredictor *ocrpredict=nullptr;
        OcrPredictor* ocrpredict_back = nullptr;

        CharacterDefectDetector *charact_detect = nullptr;
        CharacterDefectDetector* charact_detect_back = nullptr;
        PlugMeasurement* plug_m = nullptr;

        PlugMeasurement* plug_m_right = nullptr;
        PlugMeasurement* plug_m_left = nullptr;


        CharacterLocation* charact_location = nullptr;
        CharacterLocationBack* charact_location_back = nullptr;

    /// </summary>

        std::string onnxModelPath;
        string model_path_list;
        string cls_file_path;
        string rec_dict_dir;
        string model_path_list_back;
        string cls_file_path_back;
        string det_model_dir;
        string rec_model_dir;
        string det_model_dir_back;
        string rec_model_dir_back;
        string rec_dict_dir_back;

        // 阈值参数和算法参数的结构体
        OCRConfiguration config;

        DetNet_config det_config;
        RecNet_config rec_config;
        DetNet_config det_config_back;
        RecNet_config rec_config_back;
        CharactLocCustomParam custom_param;
        CharactLocCustomParamBack custom_param_back;

        PlugMeasureConfig plug_measure_config;

        singleSegALgParam segAlgParam;//字符分割参数1-8
        Configuration defect_config;
        YOLOv5* firedetmodel;
        std::vector<cv::Mat> frames;
        std::vector<std::vector<BoxInfo>> output;

        //=================================================================================================
        ErrorStatus NG_ironcladScratch_es;
        ErrorStatus NG_ironcladShinyMark_es;
        std::map<int, ErrorStatus> es_map;

        std::string vectors2string_string(std::vector<BoxInfo> vectors);
        std::string vectors2string_string(std::vector<int> vectors);
        std::string vectors2string_string(std::vector<TypeOcrStatus> vectors);
        bool initResponseAndVisual();

        bool generateResponseAndVisual(const std::vector<TypeOcrStatus>& error_res, OCRErrorValue & ocr_error_info, wikky_algo::SingleMat& data, std::string& ocr_content);

        std::map<std::string, cv::Scalar> colorMap;

        bool isNull = false;
        std::string hostname;
        std::string TONG_HOSTNAME = "DESKTOP-V3S8LSQ";
        int input_face = -1;
        double scratch_grade_thld = 60; // 50; 75 



        //respose and visualization.
        std::string result = "FAIL";;  // response relative vars
        std::string ocr_content = "";
        std::string snfrom = "";
        std::string cell_num = "";
        std::string ERCODE;
        std::string ERMSG;
        std::string DEFECT_ERCODE;
        std::string DEFECT_ERMSG;
        cv::Scalar mark_color;

        float metricConverCoeff = 2.6f;

        float ironcladScratch_filteredsize = 1.0f;

        //=================================================================================================
        CvxText* vnfont;



    public:
        Impl();
        ~Impl();
        bool subInitParam(const CheckParam& tmp);
        bool initAlgoparam(std::string& camserial);
        bool popCameraDlg(void* parent);
        bool readAlgoParam();
        bool saveAlgoParam();
        bool setLogLevel(int);
        int doing2(wikky_algo::SingleMat& data, wikky_algo::CheckParam* _checkparam = nullptr);
        int doing(wikky_algo::SingleMat& data, wikky_algo::CheckParam* m_checkparam = nullptr);

    };
}  // namespace wikky_algo


#endif // PUTTEXT_H_