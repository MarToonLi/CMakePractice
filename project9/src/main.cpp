#include "main.h"




//bool subInitParam(const CheckParam& tmp)
//{
//	// 3 init model config object with yaml params.
//	det_config.modelpath_det = det_model_dir;
//	det_config.use_gpu_det = static_cast<bool>(tmp.use_gpu_det);
//	det_config.use_dilation = static_cast<bool>(tmp.use_dilation);
//	det_config.det_db_thresh = tmp.det_db_thresh;
//	det_config.det_db_box_thresh = tmp.det_db_box_thresh;
//	det_config.det_db_unclip_ratio = tmp.det_db_unclip_ratio;
//
//
//	rec_config.dictFile = rec_dict_dir;
//	rec_config.modelpath_rec = rec_model_dir;
//	rec_config.use_gpu_rec = static_cast<bool>(tmp.use_gpu_rec);
//
//
//
//	det_config_back.modelpath_det = det_model_dir_back;
//	det_config_back.use_gpu_det = static_cast<bool>(tmp.use_gpu_det_back);
//	det_config_back.use_dilation = static_cast<bool>(tmp.use_dilation_back);
//	det_config_back.det_db_thresh = tmp.det_db_thresh_back;
//	det_config_back.det_db_box_thresh = tmp.det_db_box_thresh_back;
//	det_config_back.det_db_unclip_ratio = tmp.det_db_unclip_ratio_back;
//
//
//	rec_config_back.dictFile = rec_dict_dir_back;
//	rec_config_back.modelpath_rec = rec_model_dir_back;
//	rec_config_back.use_gpu_rec = static_cast<bool>(tmp.use_gpu_rec_back);
//
//
//
//	custom_param.left_length_low = tmp.left_length_low;
//	custom_param.left_length_high = tmp.left_length_high;
//	custom_param.right_length_low = tmp.right_length_low;
//	custom_param.right_length_high = tmp.right_length_high;
//	custom_param.first_line_height_low = tmp.first_line_height_low;
//	custom_param.first_line_height_high = tmp.first_line_height_high;
//	custom_param.second_line_height_low = tmp.second_line_height_low;
//	custom_param.second_line_height_high = tmp.second_line_height_high;
//
//
//
//	plug_measure_config.thld_value = tmp.thld_value;
//	plug_measure_config.rect_left_top = cv::Rect(tmp.lt_x, tmp.lt_y, tmp.lt_w, tmp.lt_h);
//	plug_measure_config.rect_right_top = cv::Rect(tmp.rt_x, tmp.rt_y, tmp.rt_w, tmp.rt_h);
//	plug_measure_config.rect_left_bottom = cv::Rect(tmp.lb_x, tmp.lb_y, tmp.lb_w, tmp.lb_h);
//	plug_measure_config.rect_right_bottom = cv::Rect(tmp.rb_x, tmp.rb_y, tmp.rb_w, tmp.rb_h);
//	plug_measure_config.rect_left_middle = cv::Rect(tmp.lm_x, tmp.lm_y, tmp.lm_w, tmp.lm_h);
//	plug_measure_config.rect_right_middle = cv::Rect(tmp.rm_x, tmp.rm_y, tmp.rm_w, tmp.rm_h);
//
//
//	segAlgParam.convertScaleParam = tmp.convertScaleParam;
//	segAlgParam.convertScaleDeltaParam = tmp.convertScaleDeltaParam;
//	segAlgParam.adaptiveThldBlockSize = tmp.adaptiveThldBlockSize;
//	segAlgParam.adaptiveThldSubtractValue = tmp.adaptiveThldSubtractValue;
//	segAlgParam.backfaceGrayThld = tmp.backfaceGrayThld;
//	segAlgParam.erodeEleSize = tmp.erodeEleSize;
//	segAlgParam.areaForFilter1 = tmp.areaForFilter1;
//	segAlgParam.areaForFilter2 = tmp.areaForFilter2;
//
//
//	custom_param_back.height_low = tmp.height_low;
//	custom_param_back.height_high = tmp.height_high;
//
//
//
//	config.plug_length_up = tmp.plug_length_up;
//	config.plug_length_down = tmp.plug_length_down;
//	config.plug_ratio_length = tmp.plug_ratio_length;
//	config.ratio_location = tmp.ratio_location;
//	config.ratio_location_back = tmp.ratio_location_back;
//	config.brightness_up = tmp.brightness_up;
//	config.brightness_down = tmp.brightness_down;
//	config.back_brightness_up = tmp.back_brightness_up;
//	config.back_brightness_down = tmp.back_brightness_down;
//	config.character_height_up = tmp.character_height_up;
//	config.character_height_down = tmp.character_height_down;
//	config.back_character_height_up = tmp.back_character_height_up;
//	config.back_character_height_down = tmp.back_character_height_down;
//	config.quality_score = tmp.quality_score;
//	config.quality_score_i = tmp.quality_score_i;
//	config.quality_score_l = tmp.quality_score_l;
//	config.quality_score_t = tmp.quality_score_t;
//	config.avg_score_thld_ = tmp.avg_score_thld_;
//	config.quality_score_banxing = tmp.quality_score_banxing;
//	config.quality_score_back = tmp.quality_score_back;
//	config.quality_score_i_back = tmp.quality_score_i_back;
//	config.quality_score_l_back = tmp.quality_score_l_back;
//	config.quality_score_t_back = tmp.quality_score_t_back;
//	config.avg_score_thld_back = tmp.avg_score_thld_back;
//	config.quality_score_banxing_back = tmp.quality_score_banxing_back;
//	config.rotate_thld_value = tmp.rotate_thld_value;
//
//	return false;
//}

//bool hahahamaker()
//{
//	// 2 read yaml params.
//	bool use_gpu_defect = true;
//
//	ocrpredict = new OcrPredictor(det_config, rec_config);
//	ocrpredict_back = new OcrPredictor(det_config_back, rec_config_back);
//
//	charact_detect = new CharacterDefectDetector(model_path_list, cls_file_path, use_gpu_defect, segAlgParam);
//	charact_detect_back = new CharacterDefectDetector(model_path_list_back, cls_file_path_back, use_gpu_defect, segAlgParam);
//	plug_m = new PlugMeasurement(plug_measure_config);
//	charact_location = new CharacterLocation();
//	charact_location_back = new CharacterLocationBack();
//
//	int input_face = 1;
//
//	//! ocrpredict warm up
//	int i_face = 0;        // 0-front face (两行), 1-back face(240W), 2-no character face
//	vector<tuple<string, vector<vector<int>>>> ocr_res_forOpsInit;
//	TypeOcrRes res_forOpsInit;
//	OCRErrorValue oev_forOpsInit;
//
//	LOGW("Start: the first ocr model interface.");
//	cv::Mat m = cv::imread("D:/1.bmp", 1);
//	LOGT("img: {};", m.channels());
//
//	if (m.channels() == 1) { cv::cvtColor(m, m, cv::COLOR_GRAY2BGR); }
//	cv::Mat m_rst = m.clone();
//	ocrpredict->get_input_face(m, i_face);
//	ocrpredict->run(m, i_face, ocr_res_forOpsInit);
//	LOGT("[ops init] ocrpredict init over.");
//
//	ocrpredict_back->run(m, i_face, ocr_res_forOpsInit);
//	LOGT("[ops init] ocrpredict_back init over.");
//
//	charact_detect->run(m, i_face, config.quality_score, config.quality_score_i, config.quality_score_l, config.quality_score_t, config.avg_score_thld_, config.quality_score_banxing, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit, "");
//	charact_detect->run_color_defect(m, i_face, config.brightness_up, config.brightness_down, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit);
//	charact_detect->run_character_size_defect(m, config.character_height_up, config.character_height_down, input_face, segAlgParam, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit);
//
//	charact_detect_back->run(m, i_face, config.quality_score_back, config.quality_score_i_back, config.quality_score_l_back, config.quality_score_t_back, config.avg_score_thld_back, config.quality_score_banxing_back, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit, "");
//	LOGT("[ops init] charact_detect init over.");
//
//	vector<float> output_line;
//	plug_m->run_measure(m, config.plug_length_up, config.plug_length_down, config.plug_ratio_length, output_line, res_forOpsInit, m_rst);
//	LOGT("[ops init] plug_m init over.");
//
//	charact_location->run_location(m, custom_param, config.ratio_location, output_line, ocr_res_forOpsInit, res_forOpsInit, m_rst);
//	LOGT("[ops init] charact_location init over.");
//
//	charact_location_back->run_location(m, custom_param_back, config.ratio_location_back, output_line, ocr_res_forOpsInit, res_forOpsInit, m_rst);
//	LOGT("[ops init] charact_location_back init over.");
//	LOGW("End:   the first ocr model interface.");
//	return false;
//}




int main()
{
	SetConsoleOutputCP(CP_UTF8); // allow the chinese log to show.

	//LOGD("sssssssssssssss");

	std::cout << "sss" << std::endl;

	//NA106::A106();
	//NA107::A107();
	NA108::A108();


	//yolo_min();



	// ocr
	//1 basic model files.

	//CheckParam checkparam;
	//subInitParam(checkparam);



	//hahahamaker();


	return 1;
}


/*
每次练习问题记录
A1: BGR图像转换为RGB图像
1. cv::Vec3i;



A2: BGR图像转换成灰度图
1. imgOut.at<uchar><row, col>

*/