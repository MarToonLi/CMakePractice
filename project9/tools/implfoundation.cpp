#include "foundation.h"
#include <iostream>
#include <QSettings>
#include <QCoreapplication>
#include "implfoundation.h"
#include "logger.h"
#include <functional>
#include <future>
#include <fstream>
#include <filesystem>
#include "cuda_runtime.h"
#include <windows.h>
#include "nvml.h"
#include <codecvt>

#if _MSC_VER >= 1600 //VS2015>VS>VS2010, MSVC VER= 10.0 -14.0
#pragma execution_character_set("utf-8")
#endif

namespace wikky_algo
{
	QByteArray print_enum(TypeOcrStatus e)
	{
		switch (e)
		{
		case(OK):
			return QObject::tr("合格").toLocal8Bit();
		case(NG_CHARACTER_DEFECT):
			return QObject::tr("字符缺陷").toLocal8Bit();           // 单个得分&下限
		case(NG_CHARACTER_COLOR): 
			return QObject::tr("镭雕深浅缺陷").toLocal8Bit();       // 平均灰度值 & 上下限
		case(NG_CHARACTER_SIZE):
			return QObject::tr("字符尺寸异常").toLocal8Bit();
		case(NG_PLUG_LENGTH):
			return QObject::tr("Plug长度缺陷").toLocal8Bit();
		case(NG_LOCATION):
			return QObject::tr("字符位置缺陷").toLocal8Bit(); 
		case(NG_SN_COMPARE_ERROR):
			return QObject::tr("SN与镭雕内容不一致").toLocal8Bit();  // 图片底部： app-sn: 
		case(NG_LINES_NUM):
			return QObject::tr("字符行数缺陷").toLocal8Bit();
		case(NG_CHARACTER_NUM):
			return QObject::tr("字符个数错误").toLocal8Bit();
		case(WR_SN_UNREAD_APP):
			return QObject::tr("APP未读码").toLocal8Bit();
		case(NG_DEFECT_SEGMENT):
			return QObject::tr("字符分割错误").toLocal8Bit();
		case(NG_DEFECT_AVERAGE):
			return QObject::tr("镭雕重复-模糊").toLocal8Bit();        // 平均得分&下限
		case(NG_NULL):
			return QObject::tr("空图").toLocal8Bit(); 
		case(NG_UNDEFINED):
			return QObject::tr("未定义缺陷").toLocal8Bit(); 
		case(NG_ironcladScratch):
			return QObject::tr("铁壳刮伤").toLocal8Bit();
		case(NG_ironcladShinyMark):
			return QObject::tr("铁壳亮纹").toLocal8Bit();
		case(NG_COMPANY_CODE_ERROR):
			return QObject::tr("厂商代码错误").toLocal8Bit();
		default:
			return QObject::tr("未定义缺陷").toLocal8Bit();
		}
	};

	std::string vectors2string(std::vector<TypeOcrStatus> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << vectors[i];
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}
	
	std::string vectors2string(std::vector<BoxInfo> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << vectors[i].label;
			oss << "(";
			oss << vectors[i].score;
			oss << ")";
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}

	std::string vectors2string(std::vector<int> vectors) {
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << std::to_string(vectors[i]);
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}

	std::string Alg_Foundation::Impl::vectors2string_string(std::vector<BoxInfo> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::string _tmp;
		for (int i = 0; i < vectors.size(); i++)
		{
			_tmp += es_map[vectors[i].label].en_name;
			if (vectors.size() != i + 1)
			{
				_tmp += "-";
			}
		}

		return _tmp;
	}

	std::string Alg_Foundation::Impl::vectors2string_string(std::vector<int> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::string _tmp;
		for (int i = 0; i < vectors.size(); i++)
		{
			_tmp += es_map[vectors[i]].en_name;
			if (vectors.size() != i + 1)
			{
				_tmp += "-";
			}
		}

		return _tmp;
	}

	std::string Alg_Foundation::Impl::vectors2string_string(std::vector<TypeOcrStatus> vectors) {
		if (vectors.size() == 0) { return ""; }

		std::string _tmp;
		for (int i = 0; i < vectors.size(); i++)
		{
			_tmp += print_enum_EN(vectors[i]);
			if (vectors.size() != i + 1)
			{
				_tmp += "-";
			}
		}

		return _tmp;
	}

	std::string addressToString(void* address) {
		std::ostringstream oss;
		oss << std::showbase << std::hex << reinterpret_cast<std::uintptr_t>(address);
		return oss.str();
	}

	bool outputGPU()
	{
		try
		{

		nvmlReturn_t result0;
		unsigned int device_count;
		result0 = nvmlInit();

		result0 = nvmlDeviceGetCount(&device_count);
		if (NVML_SUCCESS != result0)
		{
			std::cout << "Failed to query device count: " << nvmlErrorString(result0);

			return false;
		}
		std::cout << "Found" << device_count << " device" << std::endl;
		std::stringstream availGPUTxt;

		for (int i = 0; i < device_count; i++)
		{
			nvmlDevice_t device;
			nvmlPciInfo_t pci;
			result0 = nvmlDeviceGetHandleByIndex(i, &device);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get device count: " << nvmlErrorString(result0);
				return false;
			}

			char name[NVML_DEVICE_NAME_BUFFER_SIZE];
			result0 = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get device name: " << nvmlErrorString(result0);
				return false;
			}

			nvmlUtilization_t utilization;
			result0 = nvmlDeviceGetUtilizationRates(device, &utilization);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get utilization rates: " << nvmlErrorString(result0);
				return false;
			}

			nvmlMemory_t memory;
			result0 = nvmlDeviceGetMemoryInfo(device, &memory);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get memory info: " << nvmlErrorString(result0);
				return false;
			}

			double total = (float)(memory.total) / 1024.0f / 1024.0f / 1024.0f;
			double isused = (float)(memory.free) / 1024.0f / 1024.0f / 1024.0f;

			LOGD("isused / total =>  {}G / {}G;", isused, total);

		}
		}
		catch (...)
		{
			LOGE("GPU error;");
			return false;
		}

		return true;
	}

	void Alg_Foundation::Impl::updateparamfromdlg(CheckParam _param)
	{
		m_checkparam = _param;
		subInitParam(m_checkparam);
		float _tmp = m_checkparam.brightness_down;
		LOGD("[test]: brightness_down: {};", _tmp);
		LOGW("[Param Update] m_checkparam has been update and related params' structure also has been update.");

		void* _tmp_ptr = static_cast<void*>(&m_checkparam);
		LOGW("[Param Update] m_checkparam id: {};", addressToString(_tmp_ptr));

	}
	
	double Alg_Foundation::Impl::calBoxArea(BoxInfo boxinfo)
	{
		double boxWidth = std::abs(boxinfo.x2 - boxinfo.x1);
		double boxHeight = std::abs(boxinfo.y2 - boxinfo.y1);

		double boxArea = boxWidth * boxHeight;
		return boxArea;
	}

	double Alg_Foundation::Impl::calBoxDiagonalLength(BoxInfo boxinfo)
	{
		double boxWidth = std::abs(boxinfo.x2 - boxinfo.x1);
		double boxHeight = std::abs(boxinfo.y2 - boxinfo.y1);

		double boxDiagonalLength = std::sqrt(boxWidth * boxWidth + boxHeight * boxHeight);

		return boxDiagonalLength;
	}

	bool Alg_Foundation::Impl::defectAccumJudge(const cv::Mat& image, const std::vector<std::vector<BoxInfo>> output, ErrorStatus NG_STATUS, bool& isdefect, float accumStandardVal, cv::Mat main_area_mask)
	{
		LOGI("[defectAccumJudge]: en:{}", NG_STATUS.en_name);
		std::vector<BoxInfo> defect_outputs;
		double defect_box_val = 0;
		double defect_box_depth_val = 0;
		isdefect = false;
		double standard_val = 0;

		cv::Mat white_label_mask = cv::Mat::zeros(image.size(), CV_8UC1);

		// Damage判断是否超过阈值
		for (int i = 0; i < output[0].size(); ++i)
		{
			BoxInfo cur_box = output[0][i];
			cv::Point _tmp_pt1 = cv::Point(int(cur_box.x1), int(cur_box.y1));
			cv::Point _tmp_pt2 = cv::Point(int(cur_box.x2), int(cur_box.y2));
			cv::Rect defect_rect = cv::Rect(_tmp_pt1, _tmp_pt2);
			if (cur_box.x1 < 0 || cur_box.y1 < 0 || cur_box.x2 < 0 || cur_box.y2 < 0) { continue; }

			white_label_mask = 0;
			cv::rectangle(white_label_mask, defect_rect, cv::Scalar(255), -1);
			cv::Mat _tmp_result;
			cv::bitwise_and(white_label_mask, main_area_mask, _tmp_result);
			if (cv::countNonZero(_tmp_result) == 0) { continue; }

			if (cur_box.label == NG_STATUS.model_index) { defect_outputs.push_back(cur_box); }
		}
		LOGD("defect_outputs.size(): {}:", defect_outputs.size());

		for (int i = 0; i < defect_outputs.size(); ++i)
		{
			BoxInfo cur_box = defect_outputs[i];
			if (cur_box.x1 < 0 || cur_box.y1 < 0 || cur_box.x2 < 0 || cur_box.y2 < 0) { continue; }

			cv::Point _tmp_pt1 = cv::Point(int(cur_box.x1), int(cur_box.y1));
			cv::Point _tmp_pt2 = cv::Point(int(cur_box.x2), int(cur_box.y2));
			cv::Rect defect_rect = cv::Rect(_tmp_pt1, _tmp_pt2);


			//! 1101 S1: 筛选
			cv::Mat defect_area = image(defect_rect);
			int depth_level = 1; float depth_factor = 1;
			if (NG_ironcladScratch_es.model_index == cur_box.label) {
				int distance_mean = 0;

				float ratio = 0.2f;
				cv::Rect tmp_roi;
				tmp_roi.x = defect_area.cols * ratio;
				tmp_roi.y = defect_area.rows * ratio;

				tmp_roi.height = defect_area.rows * (1 - 2 * ratio);
				tmp_roi.width = defect_area.cols * (1 - 2 * ratio);

				cv::Mat roi_defect_area = defect_area(tmp_roi);


				cv::Mat resize_defect_area;
				cv::resize(roi_defect_area, resize_defect_area, cv::Size(), 0.2, 0.2, cv::INTER_AREA);

				bool depth_result = scratch_grade(resize_defect_area, depth_level, scratch_grade_thld, distance_mean);
				//bool depth_result = scratch_grade2(roi_defect_area, depth_level, scratch_grade_thld, distance_mean);
				if (false == depth_result) { depth_level = 1; }
				//if (1 != depth_level) { depth_factor = 1000; }
				LOGW("distance_mean : {}; ", distance_mean);

				if (depth_level == 1) { LOGD("depth_level == 1, will drop!"); continue; }
			}


			// filter op: area or leng
			double normal_leng = calBoxDiagonalLength(cur_box);                                              // 第一种做法：无投影变换
			double normal_area = calBoxArea(cur_box);

			double final_leng = normal_leng;
			double final_area = normal_area;
			std::vector<double> len_area_vals = { final_leng , final_area };                                  // 最终确定特征值

			ErrorStatus cur_es = es_map[cur_box.label];                                                       // 获取特征阈值
			bool isArea = cur_es.op;
			double box_val = len_area_vals[cur_es.op];
			double singleStandardVal = cur_es.len_area_thld[cur_es.op];
			standard_val = (defect_outputs.size() == 1) ? singleStandardVal : accumStandardVal;
			LOGD("singleStandardVal: {}; accumStandardVal: {}; standard_val: {}; ", singleStandardVal, accumStandardVal, standard_val);

			if (isArea) { box_val = box_val / metricConverCoeff; }
			double box_depth_val = box_val * depth_factor;

			if (box_val < tosys(ironcladScratch_filteredsize)) { continue; }

			defect_box_val += box_val;
			defect_box_depth_val += box_depth_val;

			LOGW("==> box_val: {}; defect_box_val: {}; box_depth_val: {}; defect_box_depth_val: {};", box_val, defect_box_val, box_depth_val, defect_box_depth_val);

			if (defect_box_depth_val > standard_val) { isdefect = true; LOGD("defect_box_val: {}; ({})", defect_box_val, standard_val); }
			else { LOGD("defect_box_depth_val: {}; ({});", defect_box_depth_val, standard_val); }
			// scratch accumu: toreal(defect_box_depth_val), toreal(standard_val)

		}

		
		cv::Point gray_line_p = cv::Point(1400, 1100);
		std::string gray_string = QObject::tr("accumu:(").toLocal8Bit().toStdString() + d2str(toreal(defect_box_depth_val),1) + ", " + d2str(toreal(standard_val),1) + ", " + d2str(ironcladScratch_filteredsize, 1) + ")";
		cv::putText(*imgrst_ptr, gray_string.c_str(), gray_line_p, 1, 5, colorMap["yellow"], 5);

		return false;
	}

	bool Alg_Foundation::Impl::initResponseAndVisual()
	{
		result = "FAIL";;  // response relative vars
		ocr_content = "";
		cell_num = "";
		snfrom = "";
		ERCODE = "";
		ERMSG = "";

		mark_color = colorMap["red"];
		//error_res.clear();

		return true;
	}
	
	bool Alg_Foundation::Impl::generateResponseAndVisual(const std::vector<TypeOcrStatus>& error_res, OCRErrorValue& ocr_error_info, wikky_algo::SingleMat& data, std::string& _ocr_content)
	{
		LOGT("Total error size:        {}; {};", error_res.size(), vectors2string(error_res));
		LOGT("Total error name(cn/en): {};", vectors2string_string(error_res));
		LOGD("1029: _ocr_content: {};", _ocr_content)

		// error_res 去重处理
		std::vector<TypeOcrStatus> uniqueVec;
		std::set<TypeOcrStatus> uniqueSet(error_res.begin(), error_res.end());
		std::copy(uniqueSet.begin(), uniqueSet.end(), std::back_inserter(uniqueVec));


		// response 变量处理
		result = (uniqueVec.size() != 0) ? "FAIL" : "PASS";                   // error_res 和 defect_error_res 共同决定是否NG
		bool isNull = (std::find(uniqueVec.begin(), uniqueVec.end(), NG_NULL) != uniqueVec.end());        // NULL仅取决于error_res
		if (isNull) { result = "NULL"; }

		QStringList tmp = QString::fromStdString(data.sn_fromscanner).split("-");
		snfrom = tmp.size() == 2 ? tmp[0].toStdString() : "";
		cell_num = tmp.size() == 2 ? tmp[1].toStdString() : "";

		ERCODE = vectors2string(uniqueVec);
		ERMSG = vectors2string_string(uniqueVec);

		// visual 使用
		mark_color = (isNull) ? colorMap["null"] : (result == "PASS") ? colorMap["green"] : colorMap["red"];
		cv::putText(data.imgrst, (isNull) ? "NU" : (result == "PASS") ? "OK" : "NG", cv::Point(20, 300), 3, 10, mark_color, 30);


		// 非无字符面的特殊处理内容
		if (input_face == 1 || input_face == 2)
		{
			cv::Point gray_line_p = cv::Point(500, 1100);
			double _gray_down = (input_face == 0) ? config.brightness_down : config.back_brightness_down;
			double _gray_up = (input_face == 0) ? config.brightness_up : config.back_brightness_up;

			std::string gray_string = QObject::tr("平均灰度: (").toLocal8Bit().toStdString();
			for (int i = 0; i < ocr_error_info.average_color_error.size(); i++)
			{
				gray_string += d2str(ocr_error_info.average_color_error[i], 0);
				if (i < ocr_error_info.average_color_error.size() - 1) {
					gray_string += ";";
				}

			}
			gray_string = gray_string + ") [" + QObject::tr("阈值:").toLocal8Bit().toStdString() + d2str(_gray_down, 0) + "~" + d2str(_gray_up, 0) + "]";

			wikky_algo::putTextZH(data.imgrst, gray_string.c_str(), gray_line_p, cv::Scalar(255, 0, 0), 80, "微软雅黑", false, false);

			//vnfont->putText(data.imgrst, "Tiếng Việt", gray_line_p, cv::Scalar(255, 0, 0));
			std::wstring vietnameseText = L"ệớệ张張Đây là tiếng Việt.哈哈哈中文显示ありがとうございます。";
			vnfont->putText(data.imgrst, vietnameseText.c_str(), gray_line_p, cv::Scalar(255, 0, 0));
			//vnfont->putText(data.imgrst, gray_string.c_str(), gray_line_p, cv::Scalar(255, 0, 0));
		}



		// 通用处理内容
		cv::putText(data.imgrst, "T: " + std::to_string(input_face), cv::Point(120, 500), 1, 5, mark_color, 10);
		for (size_t i = 0; i < uniqueVec.size(); i++)
		{
			cv::Point _line_p = cv::Point(120, 600 + 100 * i);
			std::string outtext = "R: " + print_enum(uniqueVec[i]);

			//! 特殊NG，需要加上特殊的标识
			std::string update_outtext;
			if (uniqueVec[i] == NG_CHARACTER_DEFECT) {
				update_outtext = outtext;
			}
			else if (uniqueVec[i] == NG_DEFECT_AVERAGE) {
				double _tttt = (input_face == 0) ? config.avg_score_thld_ : config.avg_score_thld_back;

				std::string _tt1 = " (";
				for (int i = 0; i < ocr_error_info.average_score_error.size(); i++)
				{
					double _tmp = ocr_error_info.average_score_error[i];
					_tt1 += d2str(_tmp, 2);

					if (i < ocr_error_info.average_score_error.size() - 1) {
						_tt1 += ";";
					}
				}
				_tt1 = _tt1 + ")" + "[" + QObject::tr("阈值: ").toLocal8Bit().toStdString() + d2str(_tttt, 2) + "]";
				update_outtext = outtext + _tt1;
			}
			else if (uniqueVec[i] == NG_CHARACTER_COLOR) {
				update_outtext = outtext;
			}
			else {
				update_outtext = outtext;
			}


			wikky_algo::putTextZH(data.imgrst, update_outtext.c_str(), _line_p, outtext.find("WR_") == std::string::npos ? mark_color : cv::Scalar(0, 255, 255), 80, "微软雅黑", false, false);

			//std::wstring wide_str(update_outtext.begin(), update_outtext.end());


			//std::wstring wide_str;

			//for (size_t i = 0; i < update_outtext.size(); ++i) {
			//	wide_str.push_back(static_cast<wchar_t>(update_outtext[i]));
			//}


			//(cv::Mat& dst, const wchar_t* wstr, cv::Point org, cv::Scalar color, int fontSize, const char* fn, bool italic, bool underline)
			std::wstring vietnameseText = L"ệớệ张張Đây là tiếng Việt.哈哈哈中文显示ありがとうございます。";
			wikky_algo::putTextWStr(data.imgrst, vietnameseText.c_str(), _line_p, outtext.find("WR_") == std::string::npos ? mark_color : cv::Scalar(0, 255, 255), 80, "微软雅黑", false, false);
		}


		cv::putText(data.imgrst, "APP-SN: " + snfrom, cv::Point(50, 1900), 3, 3, cv::Scalar(10, 192, 231), 5);


		// error_message 使用
		data.error_message.push_back(result);
		data.error_message.push_back(snfrom);
		data.error_message.push_back(_ocr_content);  //!? ocr_content 全局变量
		data.error_message.push_back(cell_num);
		data.error_message.push_back(ERCODE);
		data.error_message.push_back(ERMSG);


		LOGW("Error_message ==>        {};", vectors2string(data.error_message));
		return false;
	}
	
	Alg_Foundation::Impl::Impl()
	{
		LOGSET(SPDLOG_LEVEL_TRACE);
		LOGT("DLLInterface:{}", DLLINTERFACE);
		tid = std::this_thread::get_id();
		t = *(_Thrd_t*)(char*)&tid;
		unsigned int nId = t._Id;
		itoa(nId, buf, 10);

		bool _t_result = getHostname(hostname);
		if (_t_result) { LOGT("Hostname: {}; ", hostname); }
		else { LOGT("Failed to get hostname."); }

		clock_t st = clock();
		SetConsoleOutputCP(CP_UTF8);  // 用于设置控制台程序输出代码页。

		wikky_algo::ColorManager colorManager = wikky_algo::ColorManager();
		colorManager.initColorMap();
		colorMap = colorManager.colorMap;
	}

	Alg_Foundation::Impl::~Impl()
	{
		saveAlgoParam();
	}
	
	bool Alg_Foundation::Impl::subInitParam(const CheckParam & tmp)
	{
		NG_ironcladScratch_es   = init_errorStatus(0, "NG_ironcladScratch", "铁壳刮伤", 0, m_checkparam.ironcladScratchThreshold, m_checkparam.ironcladScratchThreshold, m_checkparam.multiIroncladScratchThreshold);
		NG_ironcladShinyMark_es = init_errorStatus(1, "NG_ironcladShinyMark", "铁壳亮纹", 1, m_checkparam.ironcladShinyMarkThreshold, m_checkparam.ironcladShinyMarkThreshold);
		es_map[NG_ironcladScratch_es.model_index]   = NG_ironcladScratch_es;
		es_map[NG_ironcladShinyMark_es.model_index] = NG_ironcladShinyMark_es;

		defect_config.modelpath = onnxModelPath;
		defect_config.confThreshold = m_checkparam.th_conf;      //? 需要修改成yaml参数
		defect_config.nmsThreshold  = m_checkparam.th_nms;
		defect_config.objThreshold  = m_checkparam.th_obj;

		LOGD("+++++++++++++++++++++++++++++++++++++++++ 缺陷检测模型参数 +++++++++++++++++++++++++++++++++++++++");
		LOGD("defect_config.modelpath:        {};", defect_config.modelpath);
		LOGD("defect_config.confThreshold:    {};", defect_config.confThreshold);
		LOGD("defect_config.nmsThreshold:     {};", defect_config.nmsThreshold);
		LOGD("defect_config.objThreshold:     {};", defect_config.objThreshold);
		LOGD("defect_config.ironcladScratchThreshold:       {};", m_checkparam.ironcladScratchThreshold);
		LOGD("defect_config.multiIroncladScratchThreshold:       {};", m_checkparam.multiIroncladScratchThreshold);
		LOGD("defect_config.ironcladShinyMarkThreshold:     {};", m_checkparam.ironcladShinyMarkThreshold);


		// 3 init model config object with yaml params.
		det_config.modelpath_det = det_model_dir;
		det_config.use_gpu_det = static_cast<bool>(tmp.use_gpu_det);
		det_config.use_dilation = static_cast<bool>(tmp.use_dilation);
		det_config.det_db_thresh = tmp.det_db_thresh;
		det_config.det_db_box_thresh = tmp.det_db_box_thresh;
		det_config.det_db_unclip_ratio = tmp.det_db_unclip_ratio;

		LOGD("\n");
		LOGD("det_config.modelpath:            {};", det_config.modelpath_det);
		LOGD("det_config.use_gpu_det:          {};", det_config.use_gpu_det);
		LOGD("det_config.use_dilation:         {};", det_config.use_dilation);
		LOGD("det_config.det_db_thresh:        {};", det_config.det_db_thresh);
		LOGD("det_config.det_db_box_thresh:    {};", det_config.det_db_box_thresh);
		LOGD("det_config.det_db_unclip_ratio:  {};", det_config.det_db_unclip_ratio);
		LOGD("\n");

		rec_config.dictFile = rec_dict_dir;
		rec_config.modelpath_rec = rec_model_dir;
		rec_config.use_gpu_rec = static_cast<bool>(tmp.use_gpu_rec);

		LOGD("rec_config.dictFile:     {};", rec_config.dictFile);
		LOGD("rec_config.modelpath:    {};", rec_config.modelpath_rec);
		LOGD("rec_config.use_gpu_rec:  {};", rec_config.use_gpu_rec);
		LOGD("\n");


		det_config_back.modelpath_det = det_model_dir_back;
		det_config_back.use_gpu_det = static_cast<bool>(tmp.use_gpu_det_back);
		det_config_back.use_dilation = static_cast<bool>(tmp.use_dilation_back);
		det_config_back.det_db_thresh = tmp.det_db_thresh_back;
		det_config_back.det_db_box_thresh = tmp.det_db_box_thresh_back;
		det_config_back.det_db_unclip_ratio = tmp.det_db_unclip_ratio_back;

		LOGD("det_config_back.modelpath:            {};", det_config_back.modelpath_det);
		LOGD("det_config_back.use_gpu_det:          {};", det_config_back.use_gpu_det);
		LOGD("det_config_back.use_dilation:         {};", det_config_back.use_dilation);
		LOGD("det_config_back.det_db_thresh:        {};", det_config_back.det_db_thresh);
		LOGD("det_config_back.det_db_box_thresh:    {};", det_config_back.det_db_box_thresh);
		LOGD("det_config_back.det_db_unclip_ratio:  {};", det_config_back.det_db_unclip_ratio);
		LOGD("\n");

		rec_config_back.dictFile = rec_dict_dir_back;
		rec_config_back.modelpath_rec = rec_model_dir_back;
		rec_config_back.use_gpu_rec = static_cast<bool>(tmp.use_gpu_rec_back);
		LOGD("rec_config_back.dictFile:     {};", rec_config_back.dictFile);
		LOGD("rec_config_back.modelpath:    {};", rec_config_back.modelpath_rec);
		LOGD("rec_config_back.use_gpu_rec:  {};", rec_config_back.use_gpu_rec);
		LOGD("\n");



		custom_param.left_length_low = tmp.left_length_low;
		custom_param.left_length_high = tmp.left_length_high;
		custom_param.right_length_low = tmp.right_length_low;
		custom_param.right_length_high = tmp.right_length_high;
		custom_param.first_line_height_low = tmp.first_line_height_low;
		custom_param.first_line_height_high = tmp.first_line_height_high;
		custom_param.second_line_height_low = tmp.second_line_height_low;
		custom_param.second_line_height_high = tmp.second_line_height_high;

		LOGD("custom_param LL: {}  {};", custom_param.left_length_low, custom_param.left_length_high);
		LOGD("custom_param RL: {}  {};", custom_param.right_length_low, custom_param.right_length_high);
		LOGD("custom_param FL: {}  {};", custom_param.first_line_height_low, custom_param.first_line_height_high);
		LOGD("custom_param SL: {}  {};", custom_param.second_line_height_low, custom_param.second_line_height_high);
		LOGD("\n");


		plug_measure_config.thld_value = tmp.thld_value;
		plug_measure_config.rect_left_top = cv::Rect(tmp.lt_x, tmp.lt_y, tmp.lt_w, tmp.lt_h);
		plug_measure_config.rect_right_top = cv::Rect(tmp.rt_x, tmp.rt_y, tmp.rt_w, tmp.rt_h);
		plug_measure_config.rect_left_bottom = cv::Rect(tmp.lb_x, tmp.lb_y, tmp.lb_w, tmp.lb_h);
		plug_measure_config.rect_right_bottom = cv::Rect(tmp.rb_x, tmp.rb_y, tmp.rb_w, tmp.rb_h);
		plug_measure_config.rect_left_middle = cv::Rect(tmp.lm_x, tmp.lm_y, tmp.lm_w, tmp.lm_h);
		plug_measure_config.rect_right_middle = cv::Rect(tmp.rm_x, tmp.rm_y, tmp.rm_w, tmp.rm_h);

		LOGD("plug_measure_config.thld_value:        {};", plug_measure_config.thld_value);
		LOGD("plug_measure_config.rect_left_top:     {};", vectors2string(plug_measure_config.rect_left_top));
		LOGD("plug_measure_config.rect_right_top:    {};", vectors2string(plug_measure_config.rect_right_top));
		LOGD("plug_measure_config.rect_left_bottom:  {};", vectors2string(plug_measure_config.rect_left_bottom));
		LOGD("plug_measure_config.rect_right_bottom: {};", vectors2string(plug_measure_config.rect_right_bottom));
		LOGD("plug_measure_config.rect_left_middle:  {};", vectors2string(plug_measure_config.rect_left_middle));
		LOGD("plug_measure_config.rect_right_middle: {};", vectors2string(plug_measure_config.rect_right_middle));
		LOGD("\n");



		segAlgParam.convertScaleParam = tmp.convertScaleParam;
		segAlgParam.convertScaleDeltaParam = tmp.convertScaleDeltaParam;
		segAlgParam.adaptiveThldBlockSize = tmp.adaptiveThldBlockSize;
		segAlgParam.adaptiveThldSubtractValue = tmp.adaptiveThldSubtractValue;
		segAlgParam.backfaceGrayThld = tmp.backfaceGrayThld;
		segAlgParam.erodeEleSize = tmp.erodeEleSize;
		segAlgParam.areaForFilter1 = tmp.areaForFilter1;
		segAlgParam.areaForFilter2 = tmp.areaForFilter2;

		LOGD("segAlgParam.convertScaleParam:           {};", segAlgParam.convertScaleParam);
		LOGD("segAlgParam.convertScaleDeltaParam:      {};", segAlgParam.convertScaleDeltaParam);
		LOGD("segAlgParam.adaptiveThldBlockSize:       {};", segAlgParam.adaptiveThldBlockSize);
		LOGD("segAlgParam.adaptiveThldSubtractValue:   {};", segAlgParam.adaptiveThldSubtractValue);
		LOGD("segAlgParam.backfaceGrayThld:            {};", segAlgParam.backfaceGrayThld);
		LOGD("segAlgParam.erodeEleSize:                {};", segAlgParam.erodeEleSize);
		LOGD("segAlgParam.areaForFilter1:              {};", segAlgParam.areaForFilter1);
		LOGD("segAlgParam.areaForFilter2:              {};", segAlgParam.areaForFilter2);
		LOGD("\n");


		custom_param_back.height_low = tmp.height_low;
		custom_param_back.height_high = tmp.height_high;

		LOGD("custom_param_back.height_low:    {};", custom_param_back.height_low);
		LOGD("custom_param_back.height_high:   {};", custom_param_back.height_high);
		LOGD("\n");



		config.plug_length_up = tmp.plug_length_up;
		config.plug_length_down = tmp.plug_length_down;
		config.plug_ratio_length = tmp.plug_ratio_length;
		config.ratio_location = tmp.ratio_location;
		config.ratio_location_back = tmp.ratio_location_back;
		config.brightness_up = tmp.brightness_up;
		config.brightness_down = tmp.brightness_down;
		config.back_brightness_up = tmp.back_brightness_up;
		config.back_brightness_down = tmp.back_brightness_down;
		config.character_height_up = tmp.character_height_up;
		config.character_height_down = tmp.character_height_down;
		config.back_character_height_up = tmp.back_character_height_up;
		config.back_character_height_down = tmp.back_character_height_down;
		config.quality_score = tmp.quality_score;
		config.quality_score_i = tmp.quality_score_i;
		config.quality_score_l = tmp.quality_score_l;
		config.quality_score_t = tmp.quality_score_t;
		config.avg_score_thld_ = tmp.avg_score_thld_;
		config.quality_score_banxing = tmp.quality_score_banxing;
		config.quality_score_back = tmp.quality_score_back;
		config.quality_score_i_back = tmp.quality_score_i_back;
		config.quality_score_l_back = tmp.quality_score_l_back;
		config.quality_score_t_back = tmp.quality_score_t_back;
		config.avg_score_thld_back = tmp.avg_score_thld_back;
		config.quality_score_banxing_back = tmp.quality_score_banxing_back;
		config.rotate_thld_value = tmp.rotate_thld_value;


		//log
		LOGD("config.plug_length_up:      {};", config.plug_length_up);
		LOGD("config.plug_length_down:    {};", config.plug_length_down);
		LOGD("config.plug_ratio_length:   {};", config.plug_ratio_length);
		LOGD("\n");

		LOGD("config.ratio_location:      {};", config.ratio_location);
		LOGD("config.ratio_location_back: {};", config.ratio_location_back);
		LOGD("\n");

		LOGD("config.brightness_up:         {};", config.brightness_up);
		LOGD("config.brightness_down:       {};", config.brightness_down);
		LOGD("config.back_brightness_up:    {};", config.back_brightness_up);
		LOGD("config.back_brightness_down:  {};", config.back_brightness_down);
		LOGD("\n");

		LOGD("config.character_height_up:         {};", config.character_height_up);
		LOGD("config.character_height_down:       {};", config.character_height_down);
		LOGD("config.back_character_height_up:    {};", config.back_character_height_up);
		LOGD("config.back_character_height_down:  {};", config.back_character_height_down);
		LOGD("\n");

		LOGD("config.quality_score:          {};", config.quality_score);
		LOGD("config.quality_score_i:        {};", config.quality_score_i);
		LOGD("config.quality_score_l:        {};", config.quality_score_l);
		LOGD("config.quality_score_t:        {};", config.quality_score_t);
		LOGD("config.avg_score_thld_:        {};", config.avg_score_thld_);
		LOGD("config.quality_score_banxing:  {};", config.quality_score_banxing);
		LOGD("\n");

		LOGD("config.quality_score_back:          {};", config.quality_score_back);
		LOGD("config.quality_score_i_back:        {};", config.quality_score_i_back);
		LOGD("config.quality_score_l_back:        {};", config.quality_score_l_back);
		LOGD("config.quality_score_t_back:        {};", config.quality_score_t_back);
		LOGD("config.avg_score_thld_back:         {};", config.avg_score_thld_back);
		LOGD("config.quality_score_banxing_back:  {};", config.quality_score_banxing_back);
		LOGD("config.rotate_thld_value:           {};", config.rotate_thld_value);
		LOGD("----------------------------------------------------------------------------------------");
		LOGD("\n");



		return false;
	}
	
	bool Alg_Foundation::Impl::initAlgoparam(std::string& _camserial)
	{
		try {

			//1 basic model files.
			model_path_list = qApp->applicationDirPath().toStdString() + "/model_files";
			cls_file_path = qApp->applicationDirPath().toStdString() + "/model_files/label.txt";
			rec_dict_dir = qApp->applicationDirPath().toStdString() + "/model_files/en_dict.txt";
			model_path_list_back = qApp->applicationDirPath().toStdString() + "/model_files_back";
			cls_file_path_back = qApp->applicationDirPath().toStdString() + "/model_files_back/label.txt";
			det_model_dir = qApp->applicationDirPath().toStdString() + "/model_files/inference_det.onnx";
			rec_model_dir = qApp->applicationDirPath().toStdString() + "/model_files/inference_rec.onnx";
			onnxModelPath = qApp->applicationDirPath().toStdString() + "/model_files_defect/3_2_best.onnx";
			det_model_dir_back = qApp->applicationDirPath().toStdString() + "/model_files_back/inference_det.onnx";
			rec_model_dir_back = qApp->applicationDirPath().toStdString() + "/model_files_back/inference_rec.onnx";
			rec_dict_dir_back = qApp->applicationDirPath().toStdString() + "/model_files_back/en_dict.txt";



			// 2 read yaml params.
			m_scamserial = _camserial;
			readAlgoParam();

			subInitParam(m_checkparam);

			outputGPU();
			//! init model operators with model config object.
			bool use_gpu_defect = true;

			ocrpredict = new OcrPredictor(det_config, rec_config);
			ocrpredict_back = new OcrPredictor(det_config_back, rec_config_back);

			charact_detect = new CharacterDefectDetector(model_path_list, cls_file_path, use_gpu_defect, segAlgParam);
			charact_detect_back = new CharacterDefectDetector(model_path_list_back, cls_file_path_back, use_gpu_defect, segAlgParam);
			plug_m = new PlugMeasurement(plug_measure_config);
			charact_location = new CharacterLocation();
			charact_location_back = new CharacterLocationBack();


			//! ocrpredict warm up
			int i_face = 0;        // 0-front face (两行), 1-back face(240W), 2-no character face
			vector<tuple<string, vector<vector<int>>>> ocr_res_forOpsInit;
			TypeOcrRes res_forOpsInit;
			OCRErrorValue oev_forOpsInit;

			bool tmp = outputRAMandGPU();
			outputGPU();
			LOGW("Start: the first ocr model interface.");
			cv::Mat m = cv::imread("D:/1.bmp", 1);
			LOGT("img: {};", m.channels());

			if (m.channels() == 1) { cv::cvtColor(m, m, cv::COLOR_GRAY2BGR); }
			cv::Mat m_rst = m.clone();
			ocrpredict->get_input_face(m, i_face);
			ocrpredict->run(m, i_face, ocr_res_forOpsInit);
			LOGT("[ops init] ocrpredict init over.");

			ocrpredict_back->run(m, i_face, ocr_res_forOpsInit);
			LOGT("[ops init] ocrpredict_back init over.");

			charact_detect->run(m, i_face, config.quality_score, config.quality_score_i, config.quality_score_l, config.quality_score_t, config.avg_score_thld_, config.quality_score_banxing, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit, "");
			charact_detect->run_color_defect(m, input_face, config.brightness_up, config.brightness_down, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit);
			charact_detect->run_character_size_defect(m, config.character_height_up, config.character_height_down, input_face, segAlgParam, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit);

			charact_detect_back->run(m, i_face, config.quality_score_back, config.quality_score_i_back, config.quality_score_l_back, config.quality_score_t_back, config.avg_score_thld_back, config.quality_score_banxing_back, ocr_res_forOpsInit, res_forOpsInit, m_rst, oev_forOpsInit, "");
			LOGT("[ops init] charact_detect init over.");

			vector<float> output_line;
			plug_m->run_measure(m, config.plug_length_up, config.plug_length_down, config.plug_ratio_length, output_line, res_forOpsInit, m_rst);
			LOGT("[ops init] plug_m init over.");

			charact_location->run_location(m, custom_param, config.ratio_location, output_line, ocr_res_forOpsInit, res_forOpsInit, m_rst);
			LOGT("[ops init] charact_location init over.");

			charact_location_back->run_location(m, custom_param_back, config.ratio_location_back, output_line, ocr_res_forOpsInit, res_forOpsInit, m_rst);
			LOGT("[ops init] charact_location_back init over.");
			LOGW("End:   the first ocr model interface.");

			outputGPU();
			// 缺陷检测模型
			firedetmodel = new YOLOv5(defect_config);
			outputGPU();

			LOGW("Start: the first defect model interface.");
			LOGD("202412181631");
			m = cv::imread("D:/1.bmp", 1);
			frames.push_back(m);
			firedetmodel->detect(frames, output, m_checkparam);
			frames.clear();
			output.clear();
			LOGW("End:   the first defect model interface.");
			outputGPU();
		}
		catch (std::exception& e)
		{
			LOGE("s: {}; may be the onnx model has problem!", e.what());
			getchar();
			return false;
		}

		return true;
	}

	bool Alg_Foundation::Impl::popCameraDlg(void* parent)
	{
		if (nullptr == algosettingdlg)
		{
			algosettingdlg = std::make_shared<Qtalgosettingdlg>((QWidget*)parent);
			algosettingdlg->SetTestCallback(std::bind(&Alg_Foundation::Impl::doing, this, std::placeholders::_1, std::placeholders::_2));
			algosettingdlg->UpdatetoalgoImpl(std::bind(&Alg_Foundation::Impl::updateparamfromdlg, this, std::placeholders::_1));
		}

		algosettingdlg->SetLastImage(lastimg);
		Param2Node(m_checkparam, m_yamlparams);
		algosettingdlg->SetLastParam(YAML::Clone(m_yamlparams));  //! 重大BUG修复之处
		algosettingdlg.get()->show();
		LOGW("popCameraDlg successfully");
		return false;
	}

	bool Alg_Foundation::Impl::readAlgoParam()
	{
		//QSettings algsetting(qApp->applicationDirPath() + "/defaultModel/" + m_scamserial.c_str() + ".ini", QSettings::IniFormat);
		//m_checkparam._iThread = algsetting.value("Default1/_Thread", 100).toInt();
		std::string _scamserial = m_scamserial;
		std::replace(_scamserial.begin(), _scamserial.end(), '/', '_');
		std::replace(_scamserial.begin(), _scamserial.end(), ':', '_');

		QString str = QString("%1/defaultModel/%2.yaml").arg(qApp->applicationDirPath()).arg(_scamserial.c_str());
		QString vnfont_str = QString("%1/defaultModel/ChillLongCangKaiShu_Regular.otf ").arg(qApp->applicationDirPath());
		//ChillLongCangKaiShu_Regular.otf             vnfont.tff   NanigoSquare-Regular.ttf
		//QString str = QString("%1/defaultModel/%2_ceshiqianduan.yaml").arg(qApp->applicationDirPath()).arg(_scamserial.c_str());
		try
		{
			m_yamlparams = YAML::LoadFile(str.toStdString());  // 读取YAML数据
			Node2Param(m_checkparam, m_yamlparams);            // 将YAML格式数据转换成预期格式
			wikky_algo::outputParam(m_checkparam);

			std::filesystem::path filepath = std::filesystem::path(str.toLocal8Bit().constData());
			LOGW("load [{}] successfully.", filepath.filename().string());
			void* _tmp_ptr = static_cast<void*>(&m_checkparam);
			LOGW("[Param Update] m_checkparam id: {};", addressToString(_tmp_ptr));
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			LOGW("load {} error: {}", str.toStdString(), e.what());
			return false;
		}


		LOGW("vnfont_str.toStdString().c_str(): {};", vnfont_str.toStdString().c_str());
		vnfont = new CvxText(vnfont_str.toStdString().c_str());

		return true;
	}

	bool Alg_Foundation::Impl::saveAlgoParam()
	{
		//QSettings algsetting(qApp->applicationDirPath() + "/defaultModel/" + m_scamserial.c_str() + ".ini", QSettings::IniFormat);
		//algsetting.setValue("Default1/_Thread", m_checkparam._iThread);

		std::string _scamserial = m_scamserial;
		std::replace(_scamserial.begin(), _scamserial.end(), '/', '_');
		std::replace(_scamserial.begin(), _scamserial.end(), ':', '_');

		QString str = QString("%1/defaultModel/%2.yaml").arg(qApp->applicationDirPath()).arg(_scamserial.c_str());
		QDir targetDir(str);
		targetDir.remove(str);

		std::ofstream fout;

		wikky_algo::outputParam(m_checkparam);
		fout.open(str.toStdString().c_str(), ios::out | ios::trunc);
		Param2Node(m_checkparam, m_yamlparams);

		try
		{
			fout << m_yamlparams;
			fout.flush();
			fout.close();
		}
		catch (YAML::ParserException e)
		{
		}
		catch (YAML::RepresentationException e)
		{
		}
		catch (YAML::Exception e)
		{
		}
		return false;
	}
	
	bool Alg_Foundation::Impl::setLogLevel(int _i)
	{
		LOGSET(_i);
		return true;
	}

	int Alg_Foundation::Impl::doing2(wikky_algo::SingleMat& data, wikky_algo::CheckParam* _checkparam)
	{
		// 路径
		std::string folder_path;
		folder_path = "D:\\Myself\\MachineVision\\resources\\HaiLun-Apple\\3-2\\3_2_UnitTestImages";

		data.error_message.push_back(" ");

		int test_epoch = 100; // 测试内存是否泄露

		// 遍历得到目录下所有的tiff文件
		try {
			for (int i = 0; i < 100; i++)
			{
				if (test_epoch != 1) { setLogLevel(SPDLOG_LEVEL_INFO); }
				for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
					if (std::filesystem::is_regular_file(entry.path())) {
						if (entry.path().filename().string().find(".png") == std::string::npos) { continue; }

						LOGW("doing: {} ({} epoch)", entry.path().filename().string(), i);
						std::string tiff_path = entry.path().string();
						cv::Mat img = cv::imread(tiff_path, cv::IMREAD_ANYDEPTH);

						wikky_algo::SingleMat data2;
						data2.imgori = img.clone();
						data2.error_message.push_back(entry.path().filename().string());
						outputGPU();
						int result = doing2(data2);
						outputGPU();
					}
				}
			}
		}
		catch (const std::exception& e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}
		return 1;
	}

	int Alg_Foundation::Impl::doing(wikky_algo::SingleMat& data, wikky_algo::CheckParam* _checkparam)
	{
		//! [init] visual 和 response 相关变量
		LOGT("\n");
		lastimg = data.imgori.clone();       //? 作为原始图像被保留？

		std::vector<tuple<std::string, std::vector<vector<int>>>> ocr_res;
		data.imgrst = data.imgori.clone();
		imgrst_ptr = & data.imgrst;
		cv::Mat ma = data.imgori.clone();
		if (data.imgori.channels() == 1)
		{
			cv::cvtColor(lastimg, ma, cv::COLOR_GRAY2RGB);
			cv::cvtColor(lastimg, data.imgrst, cv::COLOR_GRAY2BGR);
			//LOGW("test");
		}

		initResponseAndVisual();
		OCRErrorValue ocr_error_info;
		TypeOcrRes res;



		try
		{
			// [software's param test]
			if (_checkparam != nullptr)
			{
				subInitParam(*_checkparam);
				LOGW("subInitParam is taggled.");
			}

			void* _tmp_ptr = static_cast<void*>(&m_checkparam);
			//LOGW("[Param Update] m_checkparam id: {};", addressToString(_tmp_ptr));


			//! [check kong]
			cv::Rect kong_roi = cv::Rect(cv::Point(544, 1375), cv::Point(2011, 1800));
			cv::Mat gray_m;
			cv::cvtColor(ma, gray_m, cv::COLOR_RGB2GRAY);
			cv::Mat kong_image = 255 - gray_m.clone();
			cv::Mat kong_roi_img = kong_image(kong_roi);

			if (cv::countNonZero(kong_roi_img) < 100)
			{
				LOGW("there is no product  .");

				//! response.
				res.status.push_back(NG_NULL);  
				generateResponseAndVisual(res.status, ocr_error_info, data, ocr_content);   //!? 这里返回的input_face = -1;
				return -1;
			}
			// ---------------------------------------------------------------------------




			QStringList tmp = QString::fromStdString(data.sn_fromscanner).split("-");
			cell_num = tmp.size() == 2 ? tmp[1].toStdString() : "";
			snfrom = tmp.size() == 2 ? tmp[0].toStdString() : "";
			LOGD("data.sn_fromscanner: {}; snfrom: {};", data.sn_fromscanner, snfrom);


			//! [rotate image]
			cv::Mat rot_src; cv::Rect mainArea_mask_rect = cv::Rect(0, 0, ma.cols, ma.rows);
			rotate_img_before_ocr(ma, config.rotate_thld_value, rot_src, data.imgrst, mainArea_mask_rect);
			cv::Mat main_area_mask = cv::Mat::zeros(rot_src.size(), CV_8UC1);
			cv::rectangle(main_area_mask, mainArea_mask_rect, cv::Scalar(255), -1);



			//! ++++++++++++++++++++++++++++++++++++ [defect Detect.] +++++++++++++++++++++++++++++++++++++++++++++++++++++
			frames.push_back(ma);
			output.clear();
			LOGI("[defect detect] start.");
			bool return_bool = firedetmodel->detect(frames, output, m_checkparam);
			LOGI("[defect detect] over.");
			frames.clear();

			//! 模型输出结果整理
			if (return_bool == false) {
				result = "FAIL";
				res.status.emplace_back(NG_UNDEFINED);
				LOGI("[defect detect] model return false.");
			}
			else {
				cv::Scalar _tmp_color;

				//! 1008 damage缺陷区域累加，修改处2
				bool _isScratch = false;
				LOGI("[defectAccumJudge] start.");
				defectAccumJudge(gray_m, output, NG_ironcladScratch_es, _isScratch, NG_ironcladScratch_es.len_area_thld[2], main_area_mask);

				LOGI("[defect detect result postprocess] start.");
				//for (size_t i = 0; i < output.size(); ++i) 
				{
					cv::Mat white_label_mask = cv::Mat::zeros(data.imgori.size(), CV_8UC1);

					for (size_t j = 0; j < output[0].size(); ++j)
					{
						BoxInfo cur_box = output[0][j];
						if (cur_box.x1 < 0 || cur_box.y1 < 0 || cur_box.x2 < 0 || cur_box.y2 < 0) { continue; }   // 异常过滤
						
						cv::Point _tmp_pt1 = cv::Point(int(cur_box.x1), int(cur_box.y1));
						cv::Point _tmp_pt2 = cv::Point(int(cur_box.x2), int(cur_box.y2));
						LOGD("p1.x :{}; {}; {}; {};", _tmp_pt1.x, _tmp_pt1.y, _tmp_pt2.x, _tmp_pt2.y);


						//! 1101 S1: 筛选 only show
						cv::Rect defect_rect = cv::Rect(_tmp_pt1, _tmp_pt2);
						cv::Mat defect_area = gray_m(defect_rect);
						int depth_level = 1; float depth_factor = 1;
						int distance_mean = 0;

						// 过滤非主要区域的误检框
						white_label_mask = 0;
						cv::rectangle(white_label_mask, defect_rect, cv::Scalar(255), -1);
						cv::Mat _tmp_result;
						cv::bitwise_and(white_label_mask, main_area_mask, _tmp_result);
						if (cv::countNonZero(_tmp_result) == 0) { continue; }


						if (NG_ironcladScratch_es.model_index == cur_box.label) {

							float ratio = 0.2f;
							cv::Rect tmp_roi;
							tmp_roi.x = defect_area.cols * ratio;
							tmp_roi.y = defect_area.rows * ratio;

							tmp_roi.height = defect_area.rows * (1 - 2*ratio);
							tmp_roi.width = defect_area.cols * (1 - 2*ratio);


							cv::Mat roi_defect_area = defect_area(tmp_roi);
							LOGD("4");


							cv::Mat resize_defect_area;
							cv::resize(roi_defect_area, resize_defect_area, cv::Size(), 0.2, 0.2, cv::INTER_AREA);
							LOGD("5");

							bool depth_result = scratch_grade(resize_defect_area, depth_level, scratch_grade_thld, distance_mean);
							//bool depth_result = scratch_grade2(roi_defect_area, depth_level, scratch_grade_thld, distance_mean);
							LOGD("6");

							if (false == depth_result) { depth_level = 1; }
							LOGD("distance_mean: {};", distance_mean);
						}


						// size filt start
						double box_length_val = calBoxDiagonalLength(cur_box);
						double box_area_val = calBoxArea(cur_box);
						std::vector<double> len_area_vals = { box_length_val , box_area_val };

						LOGD("7");

						ErrorStatus cur_es = es_map[cur_box.label];
						bool isArea = cur_es.op;
						double box_val = len_area_vals[cur_es.op];
						double standard_val = cur_es.len_area_thld[cur_es.op];

						if (isArea) { 
							if (NG_ironcladShinyMark_es.model_index == cur_box.label) {
								box_val = box_val;
							}
							else {
								box_val = box_val / metricConverCoeff;
							}
						}
						LOGD("8");


						//! 1008 Scratch缺陷区域累加，修改处1
						if (NG_ironcladScratch_es.model_index == cur_box.label)
						{
							//if (depth_level == 1) { _tmp_color = colorMap["green"]; }
							if (box_val < tosys(ironcladScratch_filteredsize)) { _tmp_color = colorMap["green"]; }
							else if (!_isScratch) { _tmp_color = colorMap["green"]; }
							else { _tmp_color = colorMap["red"]; res.status.emplace_back(NG_ironcladScratch); }
						}
						else {
							if (box_val < standard_val) { _tmp_color = colorMap["green"]; }
							else {
								_tmp_color = colorMap["red"];
								if (cur_box.label == 0) { res.status.emplace_back(NG_ironcladScratch); }   //!? 其实有了defectAccumJudge之后，就不会再执行该句；
								if (cur_box.label == 1) { res.status.emplace_back(NG_ironcladShinyMark); }
							}  //!? 这里是与567站点不一样的地方！
						}
						LOGD("9");

						//! 显示的是box_val, 做判断的是box_depth_val;
						double show_val = -1.0;
						std::string _unit = "mm";
						if (isArea) { show_val = torealarea(box_val); _unit = "mm^2"; }
						else { show_val = toreal(box_val); _unit = "mm"; }

						LOGD("label: {}; standard_val: {}; box_val: {}; show_val: {}; (isArea: {});", cur_es.en_name, standard_val, d2str(box_val), d2str(show_val), isArea);


						cv::rectangle(data.imgrst, _tmp_pt1, _tmp_pt2, _tmp_color, 3);

						cv::Point _tmp_pt3 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 20);
						std::string outtext = cur_es.cn_name;
						putTextZH(data.imgrst, outtext.c_str(), _tmp_pt3, _tmp_color, 20, "微软雅黑", false, false);

						cv::Point _tmp_pt4 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 25);
						cv::putText(data.imgrst, "Score: " + d2str(cur_box.score), _tmp_pt4, cv::FONT_HERSHEY_TRIPLEX, 0.75, _tmp_color, 1);

						cv::Point _tmp_pt5 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 45);
						cv::putText(data.imgrst, "Size: " + d2str(show_val) + _unit, _tmp_pt5, cv::FONT_HERSHEY_TRIPLEX, 0.75, _tmp_color, 1);

						cv::Point _tmp_pt6 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 65);
						double tmp_val = (isArea) ? torealarea(standard_val) : toreal(standard_val);
						cv::putText(data.imgrst, "Size Thld: " + d2str(tmp_val) + _unit, _tmp_pt6, cv::FONT_HERSHEY_TRIPLEX, 0.75, _tmp_color, 1);

						if (NG_ironcladScratch_es.model_index == cur_box.label) {
							cv::Point _tmp_pt7 = cv::Point(int(output[0][j].x1), int(output[0][j].y1) - 85);
							cv::putText(data.imgrst, "grade: " + d2str(depth_level, 0) + "-> " + d2str(distance_mean, 2) + "|" + d2str(scratch_grade_thld,1), _tmp_pt7, cv::FONT_HERSHEY_TRIPLEX, 0.75, _tmp_color, 1);
						}
						LOGD("10");

					}
				}
				LOGI("[defect detect result postprocess] over.");

			}

			//! ----------------------------------------------------------------------------------------------






			//! ++++++++++++++++++++++++++++++++++++ [OCR Detect.] +++++++++++++++++++++++++++++++++++++++++++++++++++++
			LOGI("[ocr detect] start.");

			//! [confirm the input_face.]
			input_face = -1;                                                       // 0-front,1-back
			ocrpredict->get_input_face(rot_src, input_face);
			LOGT("[Input Face]: {}; m_scamserial: {};", input_face, m_scamserial);

			// Invalid input_face.
			if (-1 == input_face) {
				LOGE("Invalid input_face.");
				res.status.push_back(NG_UNDEFINED);
				generateResponseAndVisual(res.status, ocr_error_info, data, ocr_content);
				return -1;
			}
			// ---------------------------------------------------------------------------




			//! [ocr content.]'
			LOGI("[ocrpredict] start.");
			if (input_face == 1)
			{
				ocrpredict_back->run(rot_src, input_face, ocr_res);
			}
			else {
				ocrpredict->run(rot_src, input_face, ocr_res);
			}
			LOGI("[ocrpredict] over.");


			if (ocr_res.size() != 0) 
			{ // if there are chars.

				for (int i = 0; i < ocr_res.size(); i++)
				{
					std::tuple<std::string, std::vector<std::vector<int>>> tmp_res = ocr_res[i];
					std::string text = std::get<0>(tmp_res);
					LOGT("[OCR Contents] ({} / {}): {}", i + 1, ocr_res.size(), text);
				}


				//! [charact detect]
				//! face1: 两行字符 一面
				if (input_face == 0)
				{
					LOGI("[charact_detect] start.");
					charact_detect->run(rot_src, input_face, config.quality_score, config.quality_score_i, config.quality_score_l, config.quality_score_t, config.avg_score_thld_, config.quality_score_banxing, ocr_res, res, data.imgrst, ocr_error_info, snfrom); //!? add ocr content and char defects to res.
					LOGT("[Defect]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));


					charact_detect->run_color_defect(rot_src, input_face, config.brightness_up, config.brightness_down, ocr_res, res, data.imgrst, ocr_error_info);
					LOGT("[ Color]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));


					charact_detect->run_character_size_defect(rot_src, config.character_height_up, config.character_height_down, input_face, segAlgParam, ocr_res, res, data.imgrst, ocr_error_info);
					LOGT("[  Size]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));


					// ocr_content 赋值
					if (res.ocr_content.size() != 2) {
						res.status.emplace_back(NG_LINES_NUM);
						ocr_content = "";
					}
					else { ocr_content = res.ocr_content[1]; }
					LOGT("[response ocr_content]: {};", ocr_content);

					// ocr_content 数目
					if (ocr_content.size() != 18) { res.status.emplace_back(NG_CHARACTER_NUM); }

					LOGW("ocr_content.size: {};", ocr_content.size());

					// snfrom(240W) 比较
					if (TONG_HOSTNAME != hostname)  //!? 如果是现场使用
					{
						if (snfrom.size() == 18)
						{
							if (snfrom != ocr_content) { res.status.emplace_back(NG_SN_COMPARE_ERROR); }
						}
						else { 
							res.status.emplace_back(WR_SN_UNREAD_APP);        //? WR_SN_UNREAD_APP 仅会在0面显示！
							LOGW("[WR_SN_UNREAD_APP] snfrom: {};", snfrom); 
						}
					}
					LOGT("[ocr_content check]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));

				}
				//! face2: 240W 一面
				else if (input_face == 1)
				{
					LOGI("[charact_detect_back] start.");
					charact_detect_back->run(rot_src, input_face, config.quality_score_back, config.quality_score_i_back, config.quality_score_l_back, config.quality_score_t_back, config.avg_score_thld_back, config.quality_score_banxing_back, ocr_res, res, data.imgrst, ocr_error_info, snfrom);
					LOGT("[Defect]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));


					charact_detect_back->run_color_defect(rot_src, input_face, config.back_brightness_up, config.back_brightness_down, ocr_res, res, data.imgrst, ocr_error_info);
					LOGT("[ Color]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));


					charact_detect_back->run_character_size_defect(rot_src, config.back_character_height_up, config.back_character_height_down, input_face, segAlgParam, ocr_res, res, data.imgrst, ocr_error_info);
					LOGT("[  Size]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));


					// ocr_content 赋值
					if (res.ocr_content.size() != 1) {
						res.status.emplace_back(NG_LINES_NUM);
						ocr_content = "";
					}
					else { ocr_content = res.ocr_content[0]; }
					LOGT("[response ocr_content]: {};", ocr_content);

					// ocr_content 数目
					if (ocr_content.size() != 4) { res.status.emplace_back(NG_CHARACTER_NUM); }
					LOGW("ocr_content.size: {};", ocr_content.size());

					// snfrom(240W) 比较 
					//!? 与是否远程无关
					//if ("240W" != ocr_content) { res.status.emplace_back(NG_SN_COMPARE_ERROR); }
					LOGT("[ocr_content check]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));
				}


				//? 考虑将一下内容嵌套在上述的判断中
				LOGI("[plug_m] start.");
				vector<float> output_line;
				plug_m->run_measure(rot_src, config.plug_length_up, config.plug_length_down, config.plug_ratio_length, output_line, res, data.imgrst);
				LOGT("[  PLug]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));



				if (input_face == 0)
				{
					LOGI("[charact_location] start.");
					charact_location->run_location(rot_src, custom_param, config.ratio_location, output_line, ocr_res, res, data.imgrst);
					LOGT("[charact_location]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));
				}

				if (input_face == 1)
				{
					LOGI("[charact_location_back] start.");
					charact_location_back->run_location(rot_src, custom_param_back, config.ratio_location_back, output_line, ocr_res, res, data.imgrst);
					LOGT("[charact_location_back]: {};", (res.status.size() == 0) ? "OK" : vectors2string(res.status));
				}
			}
			//! face3: 如果没有字符，则认为是空字符的一面；
			else { 
				if (input_face == 1 || input_face == 0)
				{
					res.status.push_back(NG_CHARACTER_NUM);
				}
			}
			LOGI("[ocr detect] over.");


			//! ++++++++++++++++++++++++++++++++++++ [ feedback.] +++++++++++++++++++++++++++++++++++++++++++++++++++++
			generateResponseAndVisual(res.status, ocr_error_info,  data, ocr_content);

		}
		catch (const cv::Exception& e)
		{
			LOGW("e1: {};", e.what());
			res.status.push_back(NG_UNDEFINED);
			generateResponseAndVisual(res.status, ocr_error_info, data, ocr_content);
		}
		catch (const std::exception& e)
		{
			LOGW("e2: {};", e.what());
			res.status.push_back(NG_UNDEFINED);
			generateResponseAndVisual(res.status, ocr_error_info, data, ocr_content);
		}
		catch (...)
		{
			LOGE("e3: unkown;");
			res.status.push_back(NG_UNDEFINED);
			generateResponseAndVisual(res.status, ocr_error_info, data, ocr_content);
		}

		LOGI("[generateResponseAndVisual] over.");

		return -1;
	}
};