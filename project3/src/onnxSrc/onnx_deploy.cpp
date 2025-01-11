#include "onnx_deploy.h"



bool outputGPU()
{
	LOGD("outputGPU enter.");
	try
	{

		nvmlReturn_t result0;
		unsigned int device_count;
		result0 = nvmlInit();

		result0 = nvmlDeviceGetCount(&device_count);
		if (NVML_SUCCESS != result0)
		{
			std::cout << "Failed to query device count: " << nvmlErrorString(result0);
			return -1;
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
				return -1;
			}

			char name[NVML_DEVICE_NAME_BUFFER_SIZE];
			result0 = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get device name: " << nvmlErrorString(result0);
				return -1;
			}

			nvmlUtilization_t utilization;
			result0 = nvmlDeviceGetUtilizationRates(device, &utilization);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get utilization rates: " << nvmlErrorString(result0);
				return -1;
			}

			nvmlMemory_t memory;
			result0 = nvmlDeviceGetMemoryInfo(device, &memory);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get memory info: " << nvmlErrorString(result0);
				return -1;
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



int yolo_min()
{
	SetConsoleOutputCP(CP_UTF8); // allow the chinese log to show.
	
	std::string onnxModelPath = "F:/Projects/CMakePractice/resources/3_2_best.onnx";


	Configuration defect_config;
	YOLOv5* firedetmodel;
	std::vector<cv::Mat> frames;
	std::vector<std::vector<BoxInfo>> output;



	defect_config.modelpath = onnxModelPath;
	defect_config.confThreshold = 0.3;      //? 需要修改成yaml参数
	defect_config.nmsThreshold = 0.45;
	defect_config.objThreshold = 0.5;

	LOGD("+++++++++++++++++++++++++++++++++++++++++ 缺陷检测模型参数 +++++++++++++++++++++++++++++++++++++++");
	LOGD("defect_config.modelpath:        {};", defect_config.modelpath);
	LOGD("defect_config.confThreshold:    {};", defect_config.confThreshold);
	LOGD("defect_config.nmsThreshold:     {};", defect_config.nmsThreshold);
	LOGD("defect_config.objThreshold:     {};", defect_config.objThreshold);

	outputGPU();
	firedetmodel = new YOLOv5(defect_config);

	LOGD("202412181631");
	cv::Mat m = cv::imread("D:/58_FGJHAT005TZ000033G-1_DA3180921.png", 1);
	frames.push_back(m);
	LOGD("11");
	firedetmodel->detect(frames, output);
	LOGD("12");

	//firedetmodel->detect(frames, output);
	//output.clear();
	//LOGD("13");

	//firedetmodel->detect(frames, output);
	//LOGD("13");
	frames.clear();
	output.clear();
	LOGW("End:  the first defect model interface.");
	outputGPU();


	return 1;
}


/*
每次练习问题记录
A1: BGR图像转换为RGB图像
1. cv::Vec3i;



A2: BGR图像转换成灰度图
1. imgOut.at<uchar><row, col>

*/